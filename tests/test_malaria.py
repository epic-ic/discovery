"""The malaria example, as far as it can be checked without the study table.

Everything here that needs `msinf` is skipped when it is not importable, which
on a machine set up for `discovery` alone it is not: both projects run on the
same Python and the example's child process runs under `sys.executable` (see
`examples/malaria/fit.py`). Install `msinf` into the same environment to run
them:

    uv pip install -e /path/to/msinf
    uv run pytest tests/test_malaria.py

The study table is not shipped and is not on every machine, so the msinf tests
build a small synthetic one. That is enough for what they are for -- agreement
between two implementations of the same model, and the gates rejecting curves
that should be rejected -- neither of which is a claim about the real data.
"""

import importlib.util
import os
import sys
import textwrap
import types
from inspect import getsource
from pathlib import Path

import pytest

from discovery.model import Model

from examples.estimators import elpd_diff
from examples.malaria import __main__ as driver
from examples.malaria import data, initial

# ---------------------------------------------------------------------------
# no msinf required
# ---------------------------------------------------------------------------


#: every `(target, start)` the driver can be asked for
ALL_STARTS = sorted(initial.STARTS)


@pytest.mark.parametrize("key", ALL_STARTS)
def test_starting_models_expose_both_rewritable_regions(key):
    """Two regions, named, in the order the proposer is shown them.

    The mechanism block is named for the target, which is what makes a proposal
    for one target unusable at the other rather than silently misplaced.
    """
    target, _ = key
    assert [b.name for b in Model(initial.STARTS[key]).blocks] == ["parameters", target]


@pytest.mark.parametrize("key", ALL_STARTS)
def test_starting_models_are_syntactically_whole(key):
    compile(initial.STARTS[key], f"<{key}>", "exec")


@pytest.mark.parametrize("key", ALL_STARTS)
def test_every_start_declares_the_two_callables_the_runner_requires(key):
    """`runner.compile_provider` wants module-level `parameters` and `curves`.

    At the `immunity` target `curves` is the template's, not the candidate's, so
    this is the check that the template really provides it -- a start that only
    defined `immunity` would fail an hour into a run rather than here.
    """
    import ast

    tree = ast.parse(initial.STARTS[key])
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"parameters", "curves"} <= defined


@pytest.mark.parametrize("key", ALL_STARTS)
def test_no_start_imports_jax_directly(key):
    """The portability gate's request, checked on the models that must obey it.

    A start that reached for `jax.numpy` would be a start that fails its own
    gate, and it would teach every proposal drawn from it to do the same. Parsed
    rather than grepped: the templates name `jax.numpy` in prose, telling a
    proposer not to import it.
    """
    import ast

    imported = set()
    for node in ast.walk(ast.parse(initial.STARTS[key])):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "jax" not in imported, "starts go through `dmeq.backend().xp`"


@pytest.mark.parametrize("key", ALL_STARTS)
def test_every_start_documents_the_whole_backend_interface(key):
    """`backend().xp` alone is the wrong half to tell a proposer.

    `xp` is the arithmetic, and the arithmetic is portable for free because the
    names are shared. What a proposer cannot guess is the five primitives the array
    API does not cover -- and one of them, a recursion over the age grid, is what an
    immunity model is. A candidate told only about `xp` writes `jax.lax.scan` and
    fails a gate, or unrolls 165 classes and passes it at the cost of a fit.
    """
    from dmeq._backend import Backend

    source = initial.STARTS[key]
    for field in Backend._fields:
        assert field in source, field
    assert "backend().scan" in source
    assert "backend().set_at" in source
    assert "calculate_immunity" in source


@pytest.mark.parametrize("start", initial.START_NAMES)
def test_the_immunity_recursion_is_readable_and_fixed(start):
    """`getsource` extracts one function and not its callees.

    Left alone that makes `calculate_immunity(...)` an opaque call at the level
    that holds the actual mechanism -- section 9's complaint about `dmeq.solve`,
    one level down. So the recursion and the two-line step it is built from are
    spliced in verbatim, and *outside* the markers: a candidate may call it,
    ignore it, or write its own, but the incumbent's cannot be edited out from
    under the comparison.
    """
    from dmeq.dmeq import _next_immunity, calculate_immunity

    source = initial.STARTS["immunity", start]
    for function in (_next_immunity, calculate_immunity):
        assert getsource(function).rstrip() in source

    # outside the markers: the only rewritable regions are still the two named ones
    bodies = "\n".join(b.body for b in Model(source).blocks)
    assert "def calculate_immunity" not in bodies

    # and not imported as well, which would shadow the definition a proposer reads
    imported = {
        alias.strip()
        for line in source.splitlines()
        if line.startswith("from dmeq import ")
        for alias in line.removeprefix("from dmeq import ").split(",")
    }
    assert "calculate_immunity" not in imported


def test_the_backend_contract_cannot_break_the_docstring_it_sits_in():
    """The failure mode `undocumented` exists to prevent, on a synthetic case.

    A nested triple quote would close the candidate's own docstring early and leave
    a module that still parses and means something else -- the worst available
    failure, and one no gate downstream would catch.
    """
    quoted = 'class Thing:\n    """A docstring.\n\n    Over several lines.\n    """\n    a: int\n'
    assert initial.undocumented(quoted) == "class Thing:\n    a: int"
    with pytest.raises(ValueError, match="triple quotes survived"):
        initial.undocumented('class Thing:\n    a: int  # """ in a comment\n')


def test_a_proposal_rewriting_both_regions_splices_into_a_whole_model():
    """The thing `birthdates` never exercised: two blocks from one reply."""
    proposed = Model(initial.STARTS["curves", "scratch"]).substitute(
        {
            "parameters": "return {'a': sample('a', dist.Beta(2.0, 2.0))}",
            "curves": (
                "def mechanism(params, log_eir, ft, ages, prop, mu):\n"
                "    xp = backend().xp\n"
                "    return params['a'] * xp.ones_like(ages), xp.ones_like(ages)"
            ),
        }
    )
    compile(proposed.source, "<proposed>", "exec")
    assert "params['a']" in proposed.source
    # the scaffolding either side of the blocks survived
    assert "prev_m, inc = mechanism(params, log_eir, ft, ages, prop, mu)" in proposed.source
    assert [b.name for b in proposed.blocks] == ["parameters", "curves"]


def test_the_immunity_incumbent_is_dmeqs_own_source():
    """Item 9's whole point: a generated incumbent cannot drift from the model fitted.

    Not "agrees with" -- *is*. If this ever fails it is because someone
    reformatted the extraction, and the fix is the extraction, not a new pin.
    """
    from inspect import getsource

    from dmeq import griffin_immunity

    source = initial.STARTS["immunity", "griffin"]
    assert getsource(griffin_immunity).rstrip("\n") in source
    # and the alias that lets the fixed driver reach it
    assert "immunity = griffin_immunity" in source


def test_both_griffin_targets_sample_the_same_parameter_block():
    """The claim the two targets' numbers are compared on.

    Every name the incumbent samples is read by the immunity model and none by
    the equilibrium state solve, so the parameter block is written down once and
    reused. If it were copied, this is what would catch the copy drifting.
    """
    blocks = {
        target: {b.name: b.body for b in Model(initial.STARTS[target, "griffin"]).blocks}
        for target in initial.TARGETS
    }
    assert blocks["curves"]["parameters"] == blocks["immunity"]["parameters"]
    # and the immunity block is the same text too: at `curves` it is simply
    # followed by the state solve and the driver, inside the same region
    assert blocks["immunity"]["immunity"] in blocks["curves"]["curves"]


def test_the_target_is_an_aperture_over_one_model():
    """The property that makes the two targets' scores comparable.

    `--target` does not choose between two models. It chooses how much of one model
    the markers open up: at `immunity` the state solve and the driver are fixed text
    below the block, at `curves` the very same text is inside it. So the incumbent
    at the two targets must define exactly the same things, and differ only in which
    of them a proposal is allowed to touch.
    """
    import ast

    def defined(source):
        return {
            node.name
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }

    curves, immunity = (initial.STARTS[t, "griffin"] for t in ("curves", "immunity"))
    assert defined(curves) == defined(immunity)

    inside = {
        target: "\n".join(b.body for b in Model(initial.STARTS[target, "griffin"]).blocks)
        for target in initial.TARGETS
    }
    # the state solve and the driver: searchable at one target, fixed at the other
    for name in ("def _non_het_prev(", "def mechanism("):
        assert name in inside["curves"], name
        assert name not in inside["immunity"], name


def test_a_start_that_names_no_mechanism_for_its_target_is_refused():
    """The assembly's own error, rather than a candidate that compiles to nothing."""
    from examples.malaria.starts import curves_scratch, griffin

    with pytest.raises(ValueError, match="immunity"):
        initial.assemble("immunity", curves_scratch)
    # and `griffin` carries no mechanism at all, by design: it must be given one
    with pytest.raises(ValueError, match="curves"):
        initial.assemble("curves", griffin)


def test_split_labels_match_the_ones_msinf_writes():
    assert data.label("none") == "none"
    assert data.label("study", seed=7, n_studies=2) == "study_n2_seed7"
    with pytest.raises(ValueError):
        data.label("random_rows")


def test_a_paired_difference_is_sharper_than_either_total():
    """Why `vs best` exists, and why two totals plus their own errors will not do.

    Both candidates are scored on the same units, so a unit that is hard for one
    is hard for both. Differencing removes that shared difficulty; not
    differencing leaves it in, and it is much the larger part.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    difficulty = rng.normal(-6.0, 2.0, 35)
    a = difficulty + rng.normal(0.0, 0.3, 35)
    b = difficulty + rng.normal(-0.5, 0.3, 35)

    difference, se = elpd_diff(a, b)
    assert difference == pytest.approx(a.sum() - b.sum())
    own = min(np.sqrt(35) * a.std(ddof=1), np.sqrt(35) * b.std(ddof=1))
    assert se < own / 2


def test_candidates_scored_on_different_folds_are_refused():
    """Silence beats a number that means nothing: the units have to line up."""
    with pytest.raises(ValueError, match="same units in the same order"):
        elpd_diff([1.0, 2.0, 3.0], [1.0, 2.0])


def test_the_table_marks_a_lead_inside_its_own_error():
    """The column earns its place by what it says about the top two rows.

    A near-tie and a real gap have to read differently, or the table is still a
    ranking of numbers that cannot be ranked.
    """
    import numpy as np

    rng = np.random.default_rng(1)
    difficulty = rng.normal(-6.0, 1.5, 35)
    scored = {}
    for index, offset in ((0, 0.0), (1, -0.05), (2, -3.0)):
        pointwise = difficulty + rng.normal(offset, 0.2, 35)
        scored[index] = types.SimpleNamespace(
            index=index,
            ok=True,
            errors={},
            scores={
                "posterior.study.elpd_holdout": float(pointwise.sum()),
                "posterior.study.pointwise_holdout_incidence": tuple(pointwise[:16]),
                "posterior.study.pointwise_holdout_prevalence": tuple(pointwise[16:]),
            },
        )
    order = sorted(scored.values(), key=lambda e: -e.scores["posterior.study.elpd_holdout"])
    against = driver.leads(order, "study")

    assert against[order[0].index] == "ref"
    # the near-tie: a lead no bigger than its own error
    runner_up, _, se = against[order[1].index].partition("+-")
    assert abs(float(runner_up)) <= float(se)
    # the real gap: many times its own error
    last, _, se = against[order[2].index].partition("+-")
    assert abs(float(last)) > 4.0 * float(se)


def test_a_lead_carries_both_fits_monte_carlo_error():
    """Which mode the chains fell into moves a total by nats, which the units cannot see.

    A lead of one nat on every unit has no spread over units, so pairing alone
    calls it certain; the two fits' chain-to-chain errors are what says otherwise.
    """
    def scores(level, mcse=None):
        found = {
            "posterior.study.pointwise_holdout_incidence": (level,) * 16,
            "posterior.study.pointwise_holdout_prevalence": (level,) * 19,
        }
        return found | ({"posterior.study.elpd_holdout_mcse": mcse} if mcse is not None else {})

    difference, se = driver.compare(scores(-5.0), scores(-6.0), "study")
    assert difference == pytest.approx(35.0) and se == pytest.approx(0.0)
    difference, se = driver.compare(scores(-5.0, 3.0), scores(-6.0, 4.0), "study")
    assert difference == pytest.approx(35.0) and se == pytest.approx(5.0)
    assert driver.compare({}, scores(-6.0), "study") is None


def test_rescoring_keeps_what_the_archive_cannot_say_and_replaces_the_rest():
    """Divergences and timings describe the fit and are not archived; the scores are recomputed."""
    from examples.malaria.rescore import merge

    record = {
        "index": 3,
        "metrics": {
            "posterior": {
                "study.elpd_holdout": -190.0,
                "study.elpd_loo_train": -816.0,
                "divergences": 2.0,
                "mcmc_seconds": 6000.0,
                "study.n_divergent": 2.0,
                "study.samples.npz": {"artifact": "artifacts/0003-posterior.study.samples.npz.npz"},
                "study.fitted_train": {"artifact": "artifacts/0003-posterior.study.fitted_train.csv"},
            }
        },
    }
    rescored = {
        "metrics": {
            "study.elpd_holdout": -189.2,
            "study.elpd_loo_train": -731.5,
            "divergences": 0.0,
            "mcmc_seconds": 0.0,
            "study.n_divergent": 0.0,
            "study.chain_weights": (0.5, 0.5),
        },
        "artifacts": {},
    }
    written = {"study.fitted_train": {"artifact": "artifacts/0003-posterior.study.fitted_train.csv", "new": True}}
    merged = merge(record, rescored, ("study",), written)["metrics"]["posterior"]
    assert merged["study.elpd_holdout"] == -189.2 and merged["study.elpd_loo_train"] == -731.5
    assert merged["divergences"] == 2.0 and merged["mcmc_seconds"] == 6000.0 and merged["study.n_divergent"] == 2.0
    assert merged["study.chain_weights"] == [0.5, 0.5]
    assert merged["study.fitted_train"]["new"]
    assert merged["study.samples.npz"] == record["metrics"]["posterior"]["study.samples.npz"]


def test_the_table_pairs_on_the_split_it_ranks_on():
    """Two constants that would be a silent bug if they ever disagreed."""
    assert driver.RANK == f"posterior.{driver.RANKED_SPLIT}.elpd_holdout"
    assert driver.RANKED_SPLIT in data.SPLITS


def test_missing_inputs_are_reported_once_by_name(tmp_path):
    with pytest.raises(data.MissingData) as raised:
        data.inputs(tmp_path / "nope.csv", tmp_path / "also-nope.csv")
    assert "nope.csv" in str(raised.value)
    assert "also-nope.csv" in str(raised.value)


# ---------------------------------------------------------------------------
# msinf required
# ---------------------------------------------------------------------------

#: applied per test rather than to the module, so the tests above still run in
#: an environment that has `discovery` but not `msinf`.
needs_msinf = pytest.mark.skipif(
    importlib.util.find_spec("msinf") is None,
    reason="msinf is not importable in this interpreter",
)


@needs_msinf
def test_the_msinf_entry_points_the_runner_uses_are_importable():
    """`import msinf` is not the claim the example rests on; these are.

    `needs_msinf` asks `find_spec`, which locates the package without executing
    a single submodule -- so a dependency of msinf's that has moved an API out
    from under it still looks importable here and fails in the child process
    instead, an hour into a sweep and with the traceback in a per-candidate
    error string. These are the two modules `examples/malaria/runner.py`
    imports, imported for real. `psis_smooth` and `tail_ess` are named because
    they are the call sites that reach arviz, which is where the skew was: msinf
    and discovery must agree on one arviz-stats, and the version they agree on
    must be one whose API they both call.
    """
    import numpy as np
    from msinf.evaluate import _tail_ess, evaluate  # noqa: F401 -- the import is the test
    from msinf.fit import psis_smooth

    rng = np.random.default_rng(0)
    # Weights with an actual tail: the smoother fits a Pareto to it, and a
    # constant vector has none to fit.
    smoothed, khat = psis_smooth(rng.normal(size=256))
    assert smoothed.shape == (256,)
    assert np.isfinite(khat)

    assert _tail_ess({"x": rng.normal(size=(4, 200))}) > 0.0


@pytest.fixture(scope="module")
def f64():
    from msinf.dtypes import set_precision

    set_precision("f64")


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    """A synthetic study table and EIR prior table.

    Two studies, two sites each, incidence and microscopy prevalence in four age
    brackets.
    """
    import pandas as pd

    out = tmp_path_factory.mktemp("malaria-tables")
    bands = [(0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 20.0)]
    rows = []
    for study in (1, 2):
        for site in (10 * study, 10 * study + 1):
            for bracket, (age0, age1) in enumerate(bands, start=1):
                rows.append(dict(
                    study_index=study, site_index=site, country_name="Testland",
                    site_name=f"site{site}", type=1, age_bracket=bracket,
                    age0=age0, age1=age1, numer=20 + bracket, denom=200.0,
                    case_detection=1,
                ))
                rows.append(dict(
                    study_index=study, site_index=site, country_name="Testland",
                    site_name=f"site{site}", type=2, age_bracket=bracket,
                    age0=age0, age1=age1, numer=30 + bracket, denom=100,
                    case_detection=1,
                ))
    table = out / "study.csv"
    pd.DataFrame(rows).to_csv(table, index=False)

    eir = out / "eir.csv"
    pd.DataFrame([
        dict(site_index=s, eir_estimate=e, eir_category=c)
        for s, e, c in ((10, 5.0, 2), (11, 40.0, 3), (20, 2.0, 1), (21, 120.0, 4))
    ]).to_csv(eir, index=False)
    return table, eir


@pytest.fixture(scope="module")
def config(tables, f64):
    from msinf.config import NumpyroConfig

    table, eir = tables
    return NumpyroConfig(
        data_csv=str(table), eir_priors_csv=str(eir), prior_family="xi",
        num_warmup=1, num_samples=1, num_chains=1, collect_warmup=False, progress_bar=False,
    )


def compiled(source):
    """`(provider, namespace)`, the way `runner.fit` compiles a candidate.

    The namespace as well as the pair, because the portability gate stands in
    front of a mechanism block the fixed scaffolding calls rather than calling it
    itself -- see `gates.WATCHED`.
    """
    from examples.malaria.runner import compile_candidate, require_provider
    from msinf.curves import CurveProvider

    namespace = compile_candidate(source)
    return CurveProvider(*require_provider(namespace)), namespace


def provider_from(source):
    """Just the provider, for the tests that have no use for the namespace."""
    return compiled(source)[0]


@needs_msinf
@pytest.mark.parametrize("target", initial.TARGETS)
def test_the_incumbent_candidate_agrees_with_msinfs_own_model(config, target):
    """`TODO_discovery.md` item 4, as item 9 leaves it.

    At the `immunity` target the mechanism is `dmeq`'s own source and cannot
    drift, so half of what this used to assert is now true by construction. It is
    still worth running, and at both targets: what it pins is no longer only the
    mechanism but the hand-written parameter block, the template's unit
    convention, and -- at `immunity` -- that splicing a function into
    `solve(immunity=...)` reproduces the model msinf fits rather than merely
    something plausible.
    """
    import jax
    from msinf.model import build_model_kwargs, hierarchical_model, model_kwargs_for_sampler
    from numpyro import handlers
    from numpyro.infer.util import log_density

    reference, _, _ = build_model_kwargs(config)  # provider=None -> the incumbent
    candidate, _, _ = build_model_kwargs(
        config.__class__(**{**config.__dict__, "init_strategy": "median"}),
        provider=provider_from(initial.STARTS[target, "griffin"]),
    )

    # One parameter vector, drawn once from the reference model's own prior and
    # then held fixed for both: the two must agree at a point, not in
    # distribution.
    trace = handlers.trace(
        handlers.seed(hierarchical_model, rng_seed=jax.random.PRNGKey(0))
    ).get_trace(**model_kwargs_for_sampler(reference))
    params = {
        name: site["value"]
        for name, site in trace.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }

    densities = [
        float(log_density(hierarchical_model, (), model_kwargs_for_sampler(kwargs), params)[0])
        for kwargs in (reference, candidate)
    ]
    assert densities[0] == pytest.approx(densities[1], rel=1e-9)


@needs_msinf
@pytest.mark.parametrize("target", initial.TARGETS)
def test_the_incumbent_candidate_declares_the_same_parameters(config, target):
    """Same names, same priors -- otherwise the densities above agree by luck."""
    import jax
    from msinf.config import resolve_priors
    from msinf.curves import griffin_provider
    from numpyro import handlers

    def sites(parameters):
        trace = handlers.trace(
            handlers.seed(parameters, rng_seed=jax.random.PRNGKey(0))
        ).get_trace()
        # The distribution's family and its arguments, not its `repr`: that
        # carries an object id and would never compare equal.
        return {
            name: (
                type(site["fn"]).__name__,
                tuple(float(getattr(site["fn"], arg)) for arg in site["fn"].arg_constraints),
            )
            for name, site in trace.items()
            if site["type"] == "sample"
        }

    assert sites(provider_from(initial.STARTS[target, "griffin"]).parameters) == sites(
        griffin_provider(resolve_priors("xi")).parameters
    )


@pytest.fixture(scope="module")
def scaffolding(config, f64):
    """The fixed age grid and one site's demography, as the gates want them."""
    from msinf.age_grid import GRIFFIN_AGE_GRID
    from msinf.dtypes import asfloat
    from msinf.model import build_model_kwargs, model_kwargs_for_sampler

    kwargs = model_kwargs_for_sampler(build_model_kwargs(config)[0])
    return asfloat(GRIFFIN_AGE_GRID), kwargs["prop_site"][0], kwargs["mu_site"][0]


def gate(source, scaffolding):
    from examples.malaria import gates

    ages, prop, mu = scaffolding
    provider, namespace = compiled(source)
    return gates.check(
        provider.parameters, provider.curves,
        ages=ages, prop=prop, mu=mu, draws=4, namespace=namespace,
    )


@needs_msinf
@pytest.mark.parametrize("key", ALL_STARTS)
def test_every_starting_model_passes_the_gates(key, scaffolding):
    """A start that could not be fitted would make the whole run meaningless.

    At the `immunity` target this is also the smoke test that replaces half of
    section 4's pin: the generated module imports, splices, runs and produces
    curves a fit can be spent on.
    """
    assert gate(initial.STARTS[key], scaffolding) == (), key


BROKEN = {
    "prevalence out of range": "return 3.0 * xp.ones_like(ages), 0.1 * xp.ones_like(ages)",
    "incidence absurd": "return 0.3 * xp.ones_like(ages), 500.0 * xp.ones_like(ages)",
    "not finite": "return xp.full_like(ages, xp.nan), 0.1 * xp.ones_like(ages)",
    "no response to EIR": "return 0.3 * xp.ones_like(ages), 0.1 * xp.ones_like(ages)",
    "inverted in EIR": (
        "return 0.9 * xp.exp(-xp.exp(log_eir)) * xp.ones_like(ages), 0.1 * xp.ones_like(ages)"
    ),
}

#: the fixed shell a `BROKEN` body is dropped into: only the last line varies, so
#: each fault is one statement rather than a transcribed function.
MECHANISM = """\
def mechanism(params, log_eir, ft, ages, prop, mu):
    xp = backend().xp
    {body}"""


@needs_msinf
@pytest.mark.parametrize("fault", sorted(BROKEN))
def test_the_gates_catch_a_candidate_that_runs_but_is_nonsense(fault, scaffolding):
    """The failure mode this example has and birthdates does not.

    None of these raises, hangs or errors: each returns arrays of the right
    shape and would consume a full fit before anybody noticed.
    """
    source = Model(initial.STARTS["curves", "scratch"]).substitute(
        {"curves": MECHANISM.format(body=BROKEN[fault])}
    ).source
    assert gate(source, scaffolding), fault


#: candidates that are numerically fine and not portable, one per target. Both
#: import jax directly, which is the only way to fail this gate without also
#: failing one of the four above.
UNPORTABLE = {
    "curves": (
        "curves",
        """\
        def mechanism(params, log_eir, ft, ages, prop, mu):
            import jax.numpy as jnp
            eir = jnp.exp(log_eir)
            saturation = eir / (eir + 20.0)
            return (0.8 * saturation * jnp.ones_like(ages),
                    0.2 * saturation * jnp.ones_like(ages))""",
    ),
    "immunity": (
        "immunity",
        """\
        def immunity(eps, grid, re, p):
            import jax.numpy as jnp
            ones = jnp.ones_like(eps)
            b = p['b0'] * ones
            return Immunity(foi=b * eps, phi=p['phi0'] * ones, q=p['d1'] * ones, b=b)""",
    ),
}


@needs_msinf
@pytest.mark.parametrize("target", sorted(UNPORTABLE))
def test_the_gates_catch_a_candidate_that_only_runs_on_jax(target, scaffolding):
    """Portability as a checked property rather than a line in the brief.

    This candidate is finite, in range and rises with EIR, so every other gate
    passes it. It would fit, score, and then fail the first time anything wanted
    its curves without a jax backend -- which is the failure the error message has
    to name, so assert on the message too.
    """
    block, body = UNPORTABLE[target]
    source = Model(initial.STARTS[target, "scratch"]).substitute(
        {block: textwrap.dedent(body)}
    ).source
    reasons = gate(source, scaffolding)
    assert len(reasons) == 1, reasons
    assert "backend" in reasons[0]


@needs_msinf
def test_the_portability_gate_passes_the_starts_it_should(scaffolding):
    """The gate on its own, so a failure above says which gate failed."""
    from examples.malaria import gates

    ages, prop, mu = scaffolding
    for key in ALL_STARTS:
        provider, namespace = compiled(initial.STARTS[key])
        assert gates.portable(
            provider.parameters, provider.curves,
            ages=ages, prop=prop, mu=mu, namespace=namespace,
        ) == (), key


# ---------------------------------------------------------------------------
# the analysis bench
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench_tables(tmp_path_factory):
    """A synthetic table wide enough for the study split to draw three studies.

    The `tables` fixture has two, which is one fewer than `n_studies` defaults
    to. Five studies, two sites each, four brackets, both streams: 80 rows, of
    which the study fold leaves 32.
    """
    import pandas as pd

    out = tmp_path_factory.mktemp("malaria-bench")
    bands = [(0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 20.0)]
    rows = []
    for study in (1, 2, 3, 4, 5):
        for site in (10 * study, 10 * study + 1):
            for bracket, (age0, age1) in enumerate(bands, start=1):
                shared = dict(
                    study_index=study, site_index=site, country_name="Testland",
                    site_name=f"site{site}", age_bracket=bracket, age0=age0, age1=age1,
                    case_detection=1,
                )
                rows.append(shared | dict(type=1, numer=20 + bracket, denom=200.0))
                rows.append(shared | dict(type=2, numer=30 + bracket, denom=100))
    table = out / "study.csv"
    pd.DataFrame(rows).to_csv(table, index=False)

    sites = sorted({row["site_index"] for row in rows})
    eir = out / "eir.csv"
    pd.DataFrame([
        dict(site_index=s, eir_estimate=5.0 * i + 1.0, eir_category=(i % 5) + 1)
        for i, s in enumerate(sites)
    ]).to_csv(eir, index=False)
    return table, eir


@pytest.fixture
def bench_env(bench_tables, monkeypatch):
    """The environment `__main__` hands the pool, pointed at the synthetic table."""
    from examples.malaria import analysis

    table, eir = bench_tables
    for name, value in {
        "MALARIA_TABLE": str(table),
        "MALARIA_EIR_PRIORS": str(eir),
        "MALARIA_SPLITS": "study",
        "MALARIA_HOLDOUT_SEED": "42",
        "MALARIA_N_STUDIES": "3",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MALARIA_MORTALITY", raising=False)
    monkeypatch.setattr(analysis, "_BUILT", None)
    return analysis


@needs_msinf
def test_the_bench_hides_every_row_the_fold_holds_out(bench_env, bench_tables):
    """The property the whole module exists for.

    A row behind the `elpd_holdout` in the report is a row the proposer must not
    have measured -- and every other row is one it must still be able to see.
    """
    import pandas as pd
    from msinf.splits import Split, held_out_studies

    table, _ = bench_tables
    whole = pd.read_csv(table)
    dropped = held_out_studies(whole, Split(kind="study", n_studies=3, seed=42))
    assert len(dropped) == 3

    df, *_ = bench_env.build()
    assert not df["study_index"].isin(dropped).any()
    # and nothing else was lost: what is left is exactly the complement. The
    # whole age range included -- hiding the held-out studies is the only thing
    # the bench does, now that no fold cuts on age.
    kept = whole[~whole["study_index"].isin(dropped)]
    assert len(df) == len(kept)
    assert df["age0"].max() == kept["age0"].max()


@needs_msinf
def test_the_bench_namespace_is_what_the_description_promises(bench_env):
    """Every name the proposer is told about, and the columns it is told to read."""
    space = bench_env.namespace()
    assert set(space) == {"df", "ages", "prop", "grid", "re", "eps", "np", "pd"}
    promised = {
        "stream", "rate", "numer", "denom", "age0", "age1", "age_mid", "age_bracket",
        "eir", "eir_category", "ft", "ft_alpha", "ft_beta", "site_i", "study_index",
        "site_index", "country_name", "site_name", "case_detection",
    }
    assert promised <= set(space["df"].columns)
    assert set(space["df"]["stream"]) <= {"incidence", "prevalence"}
    assert space["prop"].shape[1] == space["ages"].size
    assert space["prop"].sum(axis=1) == pytest.approx(1.0)
    # what an `immunity` block is handed, on the same site indexing as `prop`
    assert space["re"].shape == space["eps"].shape == space["prop"].shape
    assert space["grid"].years.size == space["ages"].size
    assert space["grid"].years == pytest.approx(space["ages"])


@needs_msinf
def test_snippets_cannot_reach_each_other_through_the_frame(bench_env):
    """A fresh namespace per execution is what makes the data read-only."""
    first = bench_env.namespace()
    first["df"]["rate"] = -1.0
    first["prop"][:] = 0.0
    first["eps"][:] = 0.0
    # a NamedTuple's fields are not replaceable, but the arrays inside are writable
    first["grid"].years[:] = 0.0
    second = bench_env.namespace()
    assert (second["df"]["rate"] != -1.0).any()
    assert second["prop"].sum(axis=1) == pytest.approx(1.0)
    assert second["eps"].max() > 0.0
    assert second["grid"].years.max() > 0.0


@needs_msinf
def test_the_immunity_inputs_are_indexed_and_scaled_the_way_the_fit_scales_them(bench_env):
    """Two ways `re` and `eps` could be built wrong and still look plausible.

    Both are `(n_site, n_age)` like `prop`, so a transposed or wrongly broadcast
    build would be indexable and meaningless. The terminal age class is absorbing
    -- nobody ages out of it -- so its `re` is mortality alone, which pins the age
    axis; and `eps` is linear in the site's annual EIR, which pins the site axis.
    """
    import numpy as np

    space = bench_env.namespace()
    df, re, eps = space["df"], space["re"], space["eps"]

    # the age axis: ageing out of the absorbing class is zero, so `re` there is
    # the death rate, and strictly below what the finest infant classes see
    assert (re[:, -1] < re[:, 0]).all()
    assert re[:, -1] == pytest.approx(re[:, -1].min())

    # the site axis: two sites' eps are in the ratio of their prior EIRs
    by_site = df.groupby("site_i")["eir"].first()
    low, high = by_site.idxmin(), by_site.idxmax()
    assert eps[high] / eps[low] == pytest.approx(by_site[high] / by_site[low])
    # and the age profile is the relative biting rate: children are bitten less
    assert np.all(np.diff(eps[high]) >= 0.0)


@needs_msinf
def test_prop_is_one_row_per_site_whatever_the_mortality(bench_env, bench_tables):
    """The shape msinf's own `prop_site` has, so a snippet indexes it the same way.

    One row per canonical site in the *whole* table, not per site the training
    rows mention: `site_i` is coded over the whole table, so a narrower `prop`
    would be indexed wrong by every row that survived a study holdout.
    """
    import pandas as pd
    from msinf.data import encode_canonical_sites

    table, _ = bench_tables
    whole, _ = encode_canonical_sites(pd.read_csv(table))
    n_site = int(whole["site_i"].max()) + 1

    df, ages, prop, *_ = bench_env.build()
    assert prop.shape == (n_site, ages.size)
    assert n_site > df["site_i"].nunique()  # the study split dropped sites
    assert df["site_i"].max() < n_site
    assert prop.sum(axis=1) == pytest.approx(1.0)
    # uniform mortality: every site has the same age distribution
    assert prop.std(axis=0).max() == pytest.approx(0.0)


@needs_msinf
def test_a_wpp_mortality_table_is_resolved_the_way_the_fit_resolves_it(
    bench_env, bench_tables, tmp_path, monkeypatch
):
    """The bench weights by the demography the fit weights by, not a uniform stand-in.

    Read with msinf's own reader -- the bench copies the recursion, not the
    table handling -- and checked against `msinf.demography.resolve_demography`,
    which is what `build_model_kwargs` calls.
    """
    import numpy as np
    import pandas as pd
    from msinf.age_grid import age_grid_array
    from msinf.data import encode_canonical_sites
    from msinf.demography import resolve_demography

    table, _ = bench_tables
    frame, _ = encode_canonical_sites(pd.read_csv(table))
    ages = age_grid_array(dtype=np.float64)
    sites = sorted(pd.unique(frame["site_index"]))

    # A WPP-shaped table: mortality rising with age, and differing by site.
    rows = [
        dict(
            site_index=site,
            age_class=k,
            age_lower=float(age),
            mu_per_day=1e-4 * (1.0 + 0.05 * i) * (1.0 + (age / 50.0) ** 2),
        )
        for i, site in enumerate(sites)
        for k, age in enumerate(ages)
    ]
    mortality = tmp_path / "wpp.csv"
    pd.DataFrame(rows).to_csv(mortality, index=False)
    monkeypatch.setenv("MALARIA_MORTALITY", str(mortality))

    df, bench_ages, prop, *_ = bench_env.build()
    n_site = int(frame["site_i"].max()) + 1
    _, theirs = resolve_demography(
        df=frame, mortality_csv=mortality, n_site=n_site, ages=ages
    )
    assert prop.shape == (n_site, ages.size)
    assert np.allclose(prop, np.asarray(theirs, dtype=np.float64), rtol=1e-12, atol=0.0)
    # and it is genuinely not the uniform demography any more
    assert prop.std(axis=0).max() > 0.0


@needs_msinf
def test_the_bench_never_imports_jax(bench_tables):
    """What the whole module is arranged around.

    A subprocess, because this interpreter has almost certainly imported jax by
    the time the rest of the file has run. Four analysis workers each reserving
    gigabytes of address space for a backend nothing uses is what this buys off.

    `dmeq` is in the probe and expected to be absent for a different reason from
    the others: `build` imports it, so this asserts that importing it did not
    drag in a backend -- the property dmeq's numpy path exists to provide. The
    demography here is dmeq's own function, not a copy, and it still costs no
    jax.
    """
    import subprocess

    table, eir = bench_tables
    code = (
        "import sys\n"
        "from examples.malaria import analysis\n"
        "analysis.build()\n"
        "print(sorted(m for m in ('jax', 'jaxlib', 'dmeq', 'numpyro') if m in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=os.environ | {"MALARIA_TABLE": str(table), "MALARIA_EIR_PRIORS": str(eir)},
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().endswith("['dmeq']"), done.stdout


@needs_msinf
def test_an_absorbing_terminal_class_is_refused():
    """With no ageing out and no death, the terminal class accumulates everyone.

    dmeq's own guard, checked on the backend the bench runs it on: a traced `mu`
    under jax cannot be inspected and the check is skipped, but numpy values are
    always concrete, so in an analysis worker this always fires.
    """
    import numpy as np
    from dmeq import age_proportions, default_parameters, use_backend

    ages = np.linspace(0.0, 100.0, 20)
    mu = np.full(ages.shape, float(default_parameters()["eta"]))
    mu[-1] = 0.0
    with use_backend("numpy"):
        with pytest.raises(ValueError, match="absorbing"):
            age_proportions(ages, mu)


def test_the_arviz_stamp_is_written_so_arviz_does_not_race_to_write_it(tmp_path, monkeypatch):
    """Concurrent fits renaming arviz's one shared `daily_warning.tmp` fail on import."""
    import datetime

    from examples.malaria.runner import stamp_arviz

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    stamp_arviz()
    stamp_arviz()  # a stamp already there is overwritten, not an error

    stamp = tmp_path / "arviz" / "daily_warning"
    assert stamp.read_text() == datetime.date.today().isoformat()
    assert list(stamp.parent.iterdir()) == [stamp]
