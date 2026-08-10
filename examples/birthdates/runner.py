"""Fit one candidate model and say everything there is to say about the fit.

This runs as a child process: a JSON configuration arrives on stdin, one JSON
object leaves on stdout, and anything that goes wrong leaves on stderr with a
non-zero exit. It is the whole reason the example needs no sandbox in the
package -- model source written by a language model gets its own interpreter,
its own copy of jax, and a parent that can kill it.

Everything is computed from a single NUTS fit with a fixed budget. Identifiab-
ility, likelihood, coverage, convergence, cost and both plots are views of the
same draws, because a diagnostic recomputed from a second fit would be a
diagnostic of a different fit.
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
# all slower for the company. `set_host_device_count` appends to this below.
os.environ["XLA_FLAGS"] = (
    f"{os.environ.get('XLA_FLAGS', '')} "
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
).strip()

import matplotlib.pyplot as plt  # noqa: E402 -- after the backend is chosen
import numpy as np
from scipy.special import logsumexp

from examples.birthdates import data
from examples.estimators import columns, kl

# jax is imported inside `fit`, and not before the chain count is known: the
# number of host devices has to be set before its backend comes up.

#: what a candidate's source must leave behind
MODEL = "model"


def read_config() -> dict[str, Any]:
    """The whole configuration, before anything heavy is imported."""
    return json.loads(sys.stdin.read())


def compile_model(source: str):
    """Run the candidate's source and hand back its `model`.

    The source is a module: it does its own imports, defines its own helpers,
    and must leave a callable named `model` behind.
    """
    namespace: dict[str, Any] = {"__name__": "candidate"}
    exec(compile(source, "<candidate>", "exec"), namespace)  # noqa: S102 -- the point of the subprocess
    model = namespace.get(MODEL)
    if not callable(model):
        raise TypeError(f"the model source defines no callable {MODEL!r}")
    return model


def lppd(log_likelihood, chunk: int = 250):
    """Log mean predictive density per point, over draws, without holding them all."""
    draws = log_likelihood.shape[0]
    partial = [
        logsumexp(np.asarray(log_likelihood[start : start + chunk]), axis=0)
        for start in range(0, draws, chunk)
    ]
    return logsumexp(np.stack(partial), axis=0) - np.log(draws)


def randomised_pit(predictive, observed, rng):
    """P(Y < y) + U * P(Y = y), estimated from predictive draws.

    The observation model is discrete, so the plain PIT is discrete too and
    would look miscalibrated however good the model is; randomising within the
    jump makes it uniform under a well calibrated model. Estimating both terms
    from the draws rather than a CDF keeps this true whatever the fixed
    observation model happens to be.
    """
    below = np.mean(predictive < observed, axis=0)
    equal = np.mean(predictive == observed, axis=0)
    return below + rng.uniform(size=observed.shape) * equal


def draw_ppc(dates, observed, predictive):
    """Observed holdout days against the predictive median and 95% band."""
    low, middle, high = np.quantile(predictive, [0.025, 0.5, 0.975], axis=0)
    day = np.arange(len(observed))
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)
    for axis, window, title in (
        (axes[0], slice(None), f"held out: {dates[0]} to {dates[-1]}"),
        (axes[1], slice(0, min(366, len(observed))), f"first year: {dates[0]} on"),
    ):
        axis.fill_between(day[window], low[window], high[window], alpha=0.3, label="95% predictive")
        axis.plot(day[window], middle[window], linewidth=1.0, label="predictive median")
        axis.plot(day[window], observed[window], ".", markersize=2.0, label="observed")
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("births")
    axes[0].legend(loc="upper left", fontsize=8, ncols=3)
    axes[1].set_xlabel("day of the holdout")
    figure.suptitle("posterior predictive check", fontsize=12)
    return render(figure)


def draw_calibration(pit):
    """The PIT's empirical CDF against uniform, with a Kolmogorov-Smirnov band."""
    n = len(pit)
    ordered = np.sort(pit)
    empirical = np.arange(1, n + 1) / n
    band = 1.358 / np.sqrt(n)
    figure, axis = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    axis.fill_between([0, 1], [-band, 1 - band], [band, 1 + band], alpha=0.2, label="95% band")
    axis.plot([0, 1], [0, 1], linewidth=1.0, label="calibrated")
    axis.plot(ordered, empirical, linewidth=1.5, label="held out")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="randomised PIT", ylabel="empirical CDF")
    axis.set_title("calibration on held-out days", fontsize=12)
    axis.legend(loc="upper left", fontsize=8)
    return render(figure)


def render(figure) -> str:
    """A figure as base64 png, and closed."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()


def fit(config: dict[str, Any]) -> dict[str, Any]:
    """Sample the candidate, then score and draw the same draws."""
    import numpyro

    numpyro.set_host_device_count(config["chains"])

    import jax
    import jax.numpy as jnp
    from arviz_stats.base import array_stats
    from numpyro.handlers import seed, trace
    from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood
    from numpyro.infer.util import unconstrain_fn

    started = time.perf_counter()
    train, holdout = data.load(config["data"], config["holdout_from"])

    def arrays(series):
        return {name: jnp.asarray(np.asarray(value)) for name, value in series.features.items()}

    train_features, holdout_features = arrays(train), arrays(holdout)
    train_births = jnp.asarray(np.asarray(train.births))
    holdout_births = jnp.asarray(np.asarray(holdout.births))

    model = compile_model(config["source"])
    key = jax.random.PRNGKey(config["seed"])
    keys = jax.random.split(key, 4)

    # What the model declares, before it is asked to do anything expensive.
    sites = trace(seed(model, keys[0])).get_trace(train_features, train_births)
    latent = [
        name
        for name, site in sites.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    ]
    observed = [name for name, site in sites.items() if site.get("is_observed", False)]
    if not latent:
        raise ValueError("the model has no latent parameters to fit")
    if len(observed) != 1:
        raise ValueError(f"expected exactly one observed site, found {observed}")
    outcome = observed[0]
    if sites[outcome]["value"].shape != train_births.shape:
        raise ValueError(
            f"the observation site {outcome!r} has shape {sites[outcome]['value'].shape}, "
            f"not the {train_births.shape} of the training days"
        )
    size = int(sum(np.prod(sites[name]["value"].shape, dtype=int) for name in latent))
    if size > config["max_parameters"]:
        raise ValueError(f"{size} parameters is more than this example will fit")

    # Prior draws, kept for the identifiability comparison.
    prior = Predictive(model, num_samples=config["prior_draws"])(keys[1], train_features)
    prior = {name: prior[name] for name in latent}

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=config["warmup"],
        num_samples=config["samples"],
        num_chains=config["chains"],
        progress_bar=False,
    )
    sampling = time.perf_counter()
    mcmc.run(keys[2], train_features, train_births, extra_fields=("diverging",))
    mcmc.get_samples()[latent[0]].block_until_ready()
    mcmc_seconds = time.perf_counter() - sampling

    # Deterministic sites come back with the samples; they are consequences of
    # the parameters and not parameters, so no diagnostic here is about them.
    def parameters(samples):
        return {name: value for name, value in samples.items() if name in latent}

    posterior = parameters(mcmc.get_samples())
    by_chain = parameters(mcmc.get_samples(group_by_chain=True))
    draws = len(posterior[latent[0]])

    metrics: dict[str, float] = {}

    # (i) identifiability: did the data update every parameter, on its own scale?
    unconstrained = jax.vmap(
        lambda params: unconstrain_fn(model, (train_features, train_births), {}, params)
    )
    posterior_free = {n: np.asarray(v) for n, v in unconstrained(posterior).items()}
    prior_free = dict(columns({n: np.asarray(v) for n, v in unconstrained(prior).items()}, 1))
    divergence = {
        name: kl(column, prior_free[name]) if name in prior_free else float("nan")
        for name, column in columns(posterior_free, 1)
    }
    metrics |= {f"kl.{name}": value for name, value in divergence.items()}
    metrics["kl_min"] = float(np.nanmin(list(divergence.values())))
    metrics["unidentified"] = float(
        sum(not (value > config["identified"]) for value in divergence.values())
    )

    # (ii) likelihood: the held-out days, and the training days for comparison.
    for label, where, counts in (
        ("holdout", holdout_features, holdout_births),
        ("train", train_features, train_births),
    ):
        pointwise = lppd(log_likelihood(model, posterior, where, counts)[outcome])
        metrics[f"elpd_{label}"] = float(np.sum(pointwise))
        metrics[f"elpd_{label}_per_day"] = float(np.mean(pointwise))

    # (iii) coverage, and (the diagnostics) calibration, from held-out predictions.
    predictive = np.asarray(
        Predictive(model, posterior_samples=posterior)(keys[3], holdout_features)[outcome]
    )
    counts = np.asarray(holdout_births)
    low, high = np.quantile(predictive, [0.025, 0.975], axis=0)
    metrics["coverage95"] = float(np.mean((counts >= low) & (counts <= high)))
    metrics["coverage_error"] = metrics["coverage95"] - 0.95
    pit = randomised_pit(predictive, counts, np.random.default_rng(config["seed"]))
    metrics["pit_ks"] = float(
        np.max(np.abs(np.sort(pit) - (np.arange(1, len(pit) + 1) - 0.5) / len(pit)))
    )

    # (iv, v) convergence, per parameter.
    def per_parameter(statistic):
        """One diagnostic over every chain of every site, named coordinate by coordinate."""
        computed = {name: statistic(np.asarray(value)) for name, value in by_chain.items()}
        return {name: float(value) for name, value in columns(computed, 0)}

    rhat = per_parameter(
        lambda draws: array_stats.rhat(draws, method="rank", chain_axis=0, draw_axis=1)
    )
    ess = per_parameter(
        lambda draws: array_stats.ess(
            draws, method="tail", prob=(0.05, 0.95), chain_axis=0, draw_axis=1
        )
    )
    metrics |= {f"rhat.{name}": value for name, value in rhat.items()}
    metrics |= {f"ess_tail.{name}": value for name, value in ess.items()}
    metrics["rhat_max"] = float(np.nanmax(list(rhat.values())))
    metrics["ess_tail_min"] = float(np.nanmin(list(ess.values())))
    metrics["divergences"] = float(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
    metrics["nonfinite"] = float(
        sum(not np.isfinite(v) for v in list(rhat.values()) + list(ess.values()) + list(divergence.values()))
    )

    # (vi) what the fit cost.
    metrics["mcmc_seconds"] = mcmc_seconds
    metrics["parameters"] = float(size)
    metrics["draws"] = float(draws)
    metrics["total_seconds"] = time.perf_counter() - started

    return {
        "metrics": metrics,
        "artifacts": {
            "ppc": draw_ppc(holdout.dates, counts, predictive),
            "calibration": draw_calibration(pit),
        },
    }


def main() -> int:
    try:
        json.dump(fit(read_config()), sys.stdout, allow_nan=True)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
