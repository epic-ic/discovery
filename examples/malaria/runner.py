"""Fit one candidate curve family under every split, and score it.

A child process: a JSON configuration arrives on stdin, one JSON object leaves
on stdout, anything that goes wrong leaves on stderr with a non-zero exit. Model
source written by a language model gets its own interpreter, its own copy of jax
and a parent that can kill it.

This is a rewrite of `birthdates/runner.py` rather than a reconfiguration of it,
and the reason is that almost none of what that file does applies. There is no
observed site to hand `log_likelihood` and no `Predictive` that can replay the
model, because msinf emits two `numpyro.factor` totals with the study effects
Gauss-Hermite integrated and the site-age effect `v_jk` integrated analytically.
Everything the metrics are made of therefore comes from `msinf.evaluate` -- the
same entry point msinf's own `+experiment=holdout_*` runs go through, so a
candidate's `elpd_holdout` is comparable with the number msinf reports for the
incumbent. If it were not, the search would not be measuring what the report
claims.

What is left here is the part msinf has no opinion about: compiling a candidate,
refusing to fit one that fails the cheap gates, asking whether the data moved
each of its parameters, and putting every split in one dictionary.

Given `archive` -- a split's archived chains, by kind -- it fits nothing: the
gates already passed and the chains already ran, so it scores those chains
again (`msinf.evaluate.score`). That is how a round fitted before a change to
the scoring is brought under it without hours of sampling, and it is still done
here, in a child, because it still runs the candidate's source.

One fit per split, and `data.SPLITS` is currently one split -- the study fold.
The loop stays a loop because splits hold out different rows and so cannot share
a fit, and because their numbers are reported separately and never averaged.
"""

import base64
import io
import json
import os
import sys
import time
import traceback
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")  # nothing here has a screen to draw on

# One thread per chain, and no more. The chains are the parallelism here, and
# jax would otherwise give each of them a thread pool the size of the machine:
# with several fits in flight that is a few hundred threads on sixteen cores,
# all slower for the company. Carried over from `birthdates/runner.py`, where it
# has never actually been measured -- and it matters more here, where a fit is
# tens of minutes rather than tens of seconds.
os.environ["XLA_FLAGS"] = (
    f"{os.environ.get('XLA_FLAGS', '')} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
).strip()

import numpy as np  # noqa: E402 -- after the environment is set

from examples.estimators import columns, kl  # noqa: E402

# `gates`, `msinf` and jax are imported inside `fit`, and not before the chain
# count and the float precision are known: JAX honours neither once its backend
# has come up.

#: what a candidate's source must leave behind: the CurveProvider pair
PARAMETERS = "parameters"
CURVES = "curves"

#: parameters of the fixed observational model, sampled by the scaffolding. A
#: candidate does not declare them and is not scored on them.
SCAFFOLDING = ("rW", "rP", "log_eir", "sigma_c", "sigma_p", "alpha_c", "alpha_p", "ft_study")


def stamp_arviz() -> None:
    """Write arviz's once-a-day warning stamp before arviz can, under a name of our own.

    On its first import each day arviz writes `daily_warning.tmp` in a cache
    shared by every fit on the machine and renames it into place. Fits started
    together all write that one temporary file, and every rename after the first
    finds it gone: a `FileNotFoundError` from an import, which is what failed two
    of `runs/beat/codex-sol-r01`'s candidates without their models being run. A
    stamp already reading today makes arviz skip the write, and a per-process
    temporary name makes ours safe to race.
    """
    import datetime
    from pathlib import Path

    from platformdirs import user_cache_dir

    stamp = Path(user_cache_dir("arviz", "arviz")) / "daily_warning"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    tmp = stamp.with_name(f"{stamp.name}.{os.getpid()}.tmp")
    tmp.write_text(datetime.date.today().isoformat())
    os.replace(tmp, stamp)


def read_config() -> dict[str, Any]:
    """The whole configuration, before anything heavy is imported."""
    return json.loads(sys.stdin.read())


def compile_candidate(source: str) -> dict[str, Any]:
    """Run the candidate's source and hand back its module globals.

    The globals and not only the provider pair, because at the `immunity` target
    the block the candidate wrote is not called by anything here: `curves` is the
    template's and it reaches the candidate's `immunity` through `dmeq.solve`. The
    only way to watch what the candidate wrote is to look it up by name and put
    something in its place, which is what the portability gate does.
    """
    namespace: dict[str, Any] = {"__name__": "candidate"}
    exec(compile(source, "<candidate>", "exec"), namespace)  # noqa: S102 -- the point of the subprocess
    return namespace


def require_provider(namespace: dict[str, Any]):
    """The candidate's `(parameters, curves)`, or a `TypeError` naming what is missing.

    Two callables rather than one because `vmap` forces it -- NumPyro sample sites
    cannot be created inside the `vmap` over sites.
    """
    missing = [name for name in (PARAMETERS, CURVES) if not callable(namespace.get(name))]
    if missing:
        raise TypeError(f"the model source defines no callable {' or '.join(repr(m) for m in missing)}")
    return namespace[PARAMETERS], namespace[CURVES]


def compile_provider(source: str):
    """Compile a candidate and take its provider pair out of it."""
    return require_provider(compile_candidate(source))


def stub_curves(params, log_eir, ft, ages, prop, mu):
    """Curves that cost nothing, for the passes that only want the sample sites.

    Prior draws and the constraining transforms need the model traced, not
    solved. Under `Predictive` the real thing would mean thousands of full
    equilibrium solves at prior parameter values, many of them numerically
    hostile and none of them wanted. Swapping the provider's `curves` for this
    blocks the likelihood without restating a single sample statement, so the
    prior being drawn from is exactly the model's own -- including whatever the
    candidate invented.
    """
    del params, log_eir, ft, prop, mu
    import jax.numpy as jnp

    return jnp.zeros_like(ages), jnp.zeros_like(ages)


def candidate_sites(parameters) -> tuple[str, ...]:
    """The sample sites the candidate declares, in declaration order.

    Traced from `parameters()` alone, so the scaffolding's sites are excluded by
    construction rather than by the name list -- a candidate that happens to name
    something `sigma_c` is still its own parameter.
    """
    import jax
    from numpyro import handlers

    trace = handlers.trace(handlers.seed(parameters, rng_seed=jax.random.PRNGKey(0))).get_trace()
    return tuple(
        name for name, site in trace.items() if site["type"] == "sample" and not site.get("is_observed", False)
    )


def size_of(trace_values: dict) -> int:
    """Total scalar coordinates across a set of sites."""
    return int(sum(np.prod(np.shape(v), dtype=int) for v in trace_values.values()))


def identifiability(posterior, prior, names, model, kwargs, threshold: float) -> dict[str, float]:
    """How much the data moved each of the candidate's parameters, in nats.

    On the unconstrained scale, because that is the scale the sampler works on
    and the scale on which "the prior and the posterior are the same
    distribution" is a question about the parameter rather than about its
    support: a k-nearest-neighbour estimator run on a Beta parameter piled
    against 1 measures the boundary, not the update.

    `posterior` and `prior` carry *every* sample site, scaffolding included --
    `unconstrain_fn` re-traces the model and would try to sample anything it was
    not handed -- and `names` is what is scored. Only the candidate's own
    parameters are: `log_eir` and the study effects are the fixed observational
    model, identical for every candidate, and counting them as identified would
    flatter a model that declared nothing.
    """
    import jax
    from numpyro.infer.util import unconstrain_fn

    def free(draws):
        transformed = jax.vmap(lambda params: unconstrain_fn(model, (), kwargs, params))(draws)
        return {name: np.asarray(transformed[name]) for name in names}

    posterior_free, prior_free = free(posterior), dict(columns(free(prior), 1))
    divergence = {
        name: kl(column, prior_free[name]) if name in prior_free else float("nan")
        for name, column in columns(posterior_free, 1)
    }
    return {
        **{f"kl.{name}": value for name, value in divergence.items()},
        "kl_min": float(np.nanmin(list(divergence.values()))) if divergence else float("nan"),
        "unidentified": float(sum(not (v > threshold) for v in divergence.values())),
    }


def archived(path: str):
    """A `msinf.fit.FitResult` from a samples archive this runner wrote.

    Enough of one to score: the draws by chain and flattened. Divergences and
    timings are not archived, so a caller re-scoring keeps the originals.
    """
    from msinf.compare import BY_CHAIN_KEY
    from msinf.fit import FitResult

    loaded = np.load(path)
    by_chain = {
        name: np.asarray(loaded[name])
        for name in loaded.files
        if name != BY_CHAIN_KEY and not name.startswith("warmup__")
    }
    if BY_CHAIN_KEY in loaded.files and not bool(loaded[BY_CHAIN_KEY]):
        return FitResult(backend="nuts", draws=by_chain, seconds=0.0)
    flat = {name: value.reshape((-1,) + value.shape[2:]) for name, value in by_chain.items()}
    return FitResult(backend="nuts", draws=flat, seconds=0.0, draws_by_chain=by_chain)


def fit(config: dict[str, Any]) -> dict[str, Any]:
    """Compile, gate, then fit and score under every requested split.

    With `archive` in the config, score those archived chains instead of fitting.
    """
    import numpyro
    from msinf.dtypes import set_precision

    # Both of these fix things JAX will not honour once its backend is up, so
    # they come before the first array is built. f64 is not optional: a novel
    # curve family at f32 through two sequential scan recursions produces NaNs,
    # and the scoring pass re-runs those recursions once per draw.
    set_precision("f64")
    numpyro.set_host_device_count(config["chains"])
    stamp_arviz()  # before `msinf.evaluate` imports arviz

    import jax
    import jax.numpy as jnp
    from msinf.age_grid import GRIFFIN_AGE_GRID
    from msinf.config import NumpyroConfig
    from msinf.compare import BY_CHAIN_KEY
    from msinf.curves import CurveProvider
    from msinf.dtypes import asfloat
    from msinf.evaluate import evaluate, score
    from msinf.model import build_model_kwargs, hierarchical_model, model_kwargs_for_sampler
    from msinf.splits import Split
    from numpyro.infer import Predictive

    from examples.malaria import gates

    started = time.perf_counter()
    namespace = compile_candidate(config["source"])
    parameters, curves = require_provider(namespace)
    provider = CurveProvider(parameters=parameters, curves=curves)

    cfg = NumpyroConfig(
        data_csv=config["table"],
        eir_priors_csv=config["eir_priors"],
        mortality_csv=config["mortality"],
        num_warmup=config["warmup"],
        num_samples=config["samples"],
        num_chains=config["chains"],
        max_tree_depth=config["max_tree_depth"],
        # Every chain starts near msinf's Griffin values for each site that has
        # one -- the scaffolding always, and whatever Griffin names a candidate
        # kept -- jittered per chain, and at the prior median for the rest.
        # Shared starts keep candidates comparable with each other and with
        # msinf's own fits. The jitter is what lets chains find more than one
        # mode: this posterior has several (the Dicko Doneguebougou EIR near 50
        # or near 320, with the immunity parameters that go with each), NUTS
        # does not cross between them, and chains that all start at one point
        # all fall into one basin and report an r-hat of 1. The chains are then
        # weighted by stacking (`combine_chains`), not counted.
        init_strategy="griffin_jitter",
        prior_family=config["prior_family"],
        collect_warmup=True,
        progress_bar=False,
        seed=config["seed"],
    )

    # Built once, on the whole table: it is the same data-loading work every
    # split repeats, and the gates and the prior pass want the fixed scaffolding
    # (the age grid, the demography) rather than any particular fold.
    whole, _, _ = build_model_kwargs(cfg, provider=provider)
    sampler_kwargs = model_kwargs_for_sampler(whole)
    ages = asfloat(GRIFFIN_AGE_GRID)

    # Scalars, and the per-unit elpd vectors that the `elpd_*` totals sum.
    metrics: dict[str, float | tuple[float, ...]] = {}

    # (i) the gates, before a single second is spent sampling. A bad candidate
    # here runs and returns plausible-looking nonsense rather than raising, so
    # nothing else would catch it until the elpd came back uninterpretable.
    # Chains already archived passed them before they were sampled.
    archive = config.get("archive") or {}
    site = config["gate_site"]
    reasons = (
        []
        if archive
        else gates.check(
            parameters,
            curves,
            ages=ages,
            prop=sampler_kwargs["prop_site"][site],
            mu=sampler_kwargs["mu_site"][site],
            draws=config["gate_draws"],
            seed=config["seed"],
            inc_max=config["inc_max"],
            namespace=namespace,
        )
    )
    if reasons:
        return {"gates": list(reasons), "metrics": {}, "artifacts": {}}

    # (ii) what the candidate declared, and whether there is too much of it.
    names = candidate_sites(parameters)
    if not names:
        raise ValueError("the model's parameters() declares no sample sites; there is nothing to fit")
    prior_kwargs = sampler_kwargs | {"curves": stub_curves}
    prior = Predictive(hierarchical_model, num_samples=config["prior_draws"])(
        jax.random.PRNGKey(config["seed"]), **prior_kwargs
    )
    size = size_of({name: prior[name][0] for name in names})
    if size > config["max_parameters"]:
        raise ValueError(f"{size} candidate parameters is more than this example will fit")
    metrics["parameters"] = float(size)

    # (iii) one fit per split, and every number and picture that fit has to say.
    artifacts: dict[str, str] = {}
    worst: dict[str, list[float]] = {"rhat_max": [], "ess_tail_min": [], "divergences": [], "mcmc_seconds": []}
    for kind in config["splits"]:
        split = Split(
            kind=kind,
            seed=config["holdout_seed"],
            n_studies=config["n_studies"],
        )
        scoring = {
            "n_score_draws": config["n_score_draws"],
            "n_replicates": config["n_replicates"],
            "seed": config["seed"],
            "figures": True,
            "combine_chains": config["combine_chains"],
        }
        if kind in archive:
            kwargs, _, _ = build_model_kwargs(cfg, provider=provider, split=split)
            result = score(cfg, archived(archive[kind]), kwargs, split=split, **scoring)
        else:
            result = evaluate(cfg, provider=provider, split=split, **scoring)
        metrics |= {f"{kind}.{name}": float(v) for name, v in result.metrics.items()}
        # Training-only, like the LOO they are fitted to: which chains carry the predictive.
        metrics[f"{kind}.chain_weights"] = tuple(result.info["chain_weights"])
        # The per-unit elpd behind each total. Carried as vectors rather than
        # summarised here because the useful comparison is between candidates,
        # which this process cannot see: it fits one model and knows nothing
        # about the others. The driver holds the leaderboard and does the
        # pairing -- see `elpd_diff` in `examples/estimators.py`.
        metrics |= {
            f"{kind}.pointwise_{name}": tuple(float(x) for x in vector) for name, vector in result.pointwise.items()
        }
        artifacts |= {
            f"{kind}.{name.removesuffix('.png')}": base64.b64encode(payload).decode()
            for name, payload in result.figures.items()
        }
        # Every observed row beside its posterior predictive, per fold, from this
        # candidate's own curves. `feedback.py` hands the training one to the
        # next round's agents; the held-out one is for whoever reads the report.
        artifacts |= {
            f"{kind}.fitted_{fold}.csv": base64.b64encode(frame.to_csv(index=False).encode()).decode()
            for fold, frame in result.fitted.items()
        }

        # Match msinf's run_inference artifact format: posterior arrays are
        # grouped by chain and warmup arrays are named warmup__<site>. This is
        # written from the FitResult already used for scoring, so archiving it
        # never triggers a second fit.
        fit_result = result.fit_result
        if fit_result is not None and kind not in archive:
            samples = fit_result.draws_by_chain or fit_result.draws
            payload = io.BytesIO()
            np.savez(
                payload,
                **{name: np.asarray(value) for name, value in samples.items()},
                **fit_result.extra.get("warmup_samples", {}),
                **{BY_CHAIN_KEY: np.asarray(fit_result.draws_by_chain is not None)},
            )
            artifacts[f"{kind}.samples.npz"] = base64.b64encode(payload.getvalue()).decode()

        # Identifiability against the prior, from the draws this fit already
        # produced. A second fit to answer it would be a diagnostic of a
        # different fit.
        metrics |= {
            f"{kind}.{name}": value
            for name, value in identifiability(
                {name: jnp.asarray(value) for name, value in result.draws.items()},
                {name: jnp.asarray(value) for name, value in prior.items()},
                names,
                hierarchical_model,
                prior_kwargs,
                config["identified"],
            ).items()
        }

        worst["rhat_max"].append(result.metrics.get("max_r_hat", float("nan")))
        worst["ess_tail_min"].append(result.metrics.get("min_ess_tail", float("nan")))
        worst["divergences"].append(result.metrics.get("n_divergent", 0.0))
        worst["mcmc_seconds"].append(result.metrics.get("mcmc_seconds", float("nan")))

    # (iv) the run-level summary: the worst of the splits, not the average of
    # them. A candidate whose chains converged under one fold and not another
    # has not converged.
    with np.errstate(invalid="ignore"):
        metrics["rhat_max"] = float(np.nanmax(worst["rhat_max"]))
        metrics["ess_tail_min"] = float(np.nanmin(worst["ess_tail_min"]))
        metrics["divergences"] = float(np.nansum(worst["divergences"]))
        metrics["mcmc_seconds"] = float(np.nansum(worst["mcmc_seconds"]))
        kls = [metrics[f"{k}.kl_min"] for k in config["splits"] if f"{k}.kl_min" in metrics]
        unid = [metrics[f"{k}.unidentified"] for k in config["splits"] if f"{k}.unidentified" in metrics]
        metrics["kl_min"] = float(np.nanmin(kls)) if kls else float("nan")
        metrics["unidentified"] = float(np.nanmax(unid)) if unid else float("nan")
    metrics["total_seconds"] = time.perf_counter() - started

    return {"gates": [], "metrics": metrics, "artifacts": artifacts}


def main() -> int:
    try:
        json.dump(fit(read_config()), sys.stdout, allow_nan=True)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
