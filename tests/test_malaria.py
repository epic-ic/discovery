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
from pathlib import Path

import pytest

from discovery.model import Model

from examples.estimators import elpd_diff
from examples.malaria import __main__ as driver
from examples.malaria import data, initial

# ---------------------------------------------------------------------------
# no msinf required
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(initial.STARTS))
def test_starting_models_expose_both_rewritable_regions(name):
    """Two regions, named, in the order the proposer is shown them."""
    assert [b.name for b in Model(initial.STARTS[name]).blocks] == ["parameters", "curves"]


@pytest.mark.parametrize("name", sorted(initial.STARTS))
def test_starting_models_are_syntactically_whole(name):
    compile(initial.STARTS[name], f"<{name}>", "exec")


def test_a_proposal_rewriting_both_regions_splices_into_a_whole_model():
    """The thing `birthdates` never exercised: two blocks from one reply."""
    proposed = Model(initial.SCRATCH).substitute(
        {
            "parameters": "return {'a': sample('a', dist.Beta(2.0, 2.0))}",
            "curves": "prev_m = params['a'] * jnp.ones_like(ages)\ninc = jnp.ones_like(ages)",
        }
    )
    compile(proposed.source, "<proposed>", "exec")
    assert "params['a']" in proposed.source
    # the scaffolding either side of the blocks survived
    assert "def curves(params, log_eir, ft, ages, prop, mu):" in proposed.source
    assert [b.name for b in proposed.blocks] == ["parameters", "curves"]


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


def provider_from(source):
    """Compile a candidate's source the way `runner.compile_provider` does."""
    from examples.malaria.runner import compile_provider
    from msinf.curves import CurveProvider

    return CurveProvider(*compile_provider(source))


@needs_msinf
def test_the_incumbent_candidate_agrees_with_msinfs_own_model(config):
    """`TODO_discovery.md` item 4: the skeleton is duplicated, so pin it.

    `initial.GRIFFIN` restates the Griffin provider as source a proposer can
    read and correct. `msinf.curves.griffin_provider` is the same model as the
    inference package fits it. They will drift, and a silent drift means the
    search is optimising a model that is not the one msinf reports on -- so fix
    a seed and a parameter vector and assert the two log-densities agree.
    """
    import jax
    from msinf.model import build_model_kwargs, hierarchical_model, model_kwargs_for_sampler
    from numpyro import handlers
    from numpyro.infer.util import log_density

    reference, _, _ = build_model_kwargs(config)  # provider=None -> the incumbent
    candidate, _, _ = build_model_kwargs(
        config.__class__(**{**config.__dict__, "init_strategy": "median"}),
        provider=provider_from(initial.GRIFFIN),
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
def test_the_incumbent_candidate_declares_the_same_parameters(config):
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

    assert sites(provider_from(initial.GRIFFIN).parameters) == sites(
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
    provider = provider_from(source)
    return gates.check(provider.parameters, provider.curves, ages=ages, prop=prop, mu=mu, draws=4)


@needs_msinf
def test_both_starting_models_pass_the_gates(scaffolding):
    """A start that could not be fitted would make the whole run meaningless."""
    for name, source in initial.STARTS.items():
        assert gate(source, scaffolding) == (), name


BROKEN = {
    "prevalence out of range": "prev_m = 3.0 * jnp.ones_like(ages)\ninc = 0.1 * jnp.ones_like(ages)",
    "incidence absurd": "prev_m = 0.3 * jnp.ones_like(ages)\ninc = 500.0 * jnp.ones_like(ages)",
    "not finite": "prev_m = jnp.full_like(ages, jnp.nan)\ninc = 0.1 * jnp.ones_like(ages)",
    "no response to EIR": "prev_m = 0.3 * jnp.ones_like(ages)\ninc = 0.1 * jnp.ones_like(ages)",
    "inverted in EIR": (
        "prev_m = 0.9 * jnp.exp(-jnp.exp(log_eir)) * jnp.ones_like(ages)\n"
        "inc = 0.1 * jnp.ones_like(ages)"
    ),
}


@needs_msinf
@pytest.mark.parametrize("fault", sorted(BROKEN))
def test_the_gates_catch_a_candidate_that_runs_but_is_nonsense(fault, scaffolding):
    """The failure mode this example has and birthdates does not.

    None of these raises, hangs or errors: each returns arrays of the right
    shape and would consume a full fit before anybody noticed.
    """
    source = Model(initial.SCRATCH).substitute(
        {"curves": textwrap.dedent(BROKEN[fault])}
    ).source
    assert gate(source, scaffolding), fault


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

    df, _, _ = bench_env.build()
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
    assert set(space) == {"df", "ages", "prop", "np", "pd"}
    promised = {
        "stream", "rate", "numer", "denom", "age0", "age1", "age_mid", "age_bracket",
        "eir", "eir_category", "ft", "ft_alpha", "ft_beta", "site_i", "study_index",
        "site_index", "country_name", "site_name", "case_detection",
    }
    assert promised <= set(space["df"].columns)
    assert set(space["df"]["stream"]) <= {"incidence", "prevalence"}
    assert space["prop"].shape[1] == space["ages"].size
    assert space["prop"].sum(axis=1) == pytest.approx(1.0)


@needs_msinf
def test_snippets_cannot_reach_each_other_through_the_frame(bench_env):
    """A fresh namespace per execution is what makes the data read-only."""
    first = bench_env.namespace()
    first["df"]["rate"] = -1.0
    first["prop"][:] = 0.0
    second = bench_env.namespace()
    assert (second["df"]["rate"] != -1.0).any()
    assert second["prop"].sum(axis=1) == pytest.approx(1.0)


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

    df, ages, prop = bench_env.build()
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

    df, bench_ages, prop = bench_env.build()
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
    """The reason `age_proportions` is duplicated at all.

    A subprocess, because this interpreter has almost certainly imported jax by
    the time the rest of the file has run. Four analysis workers each reserving
    gigabytes of address space for a backend nothing uses is what this buys off.
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
    assert done.stdout.strip().endswith("[]"), done.stdout


@needs_msinf
def test_the_benchs_age_proportions_agree_with_dmeq(f64):
    """The pin on the one formula this module copies.

    A duplicated formula that nothing compares is a formula that will drift, and
    a demography that disagrees with the one the likelihood weights by would
    have an agent reasoning about a different population from the one being
    fitted. Not bitwise: the copy is a `cumprod` where dmeq scans, which is a
    different association of the same products.
    """
    import numpy as np
    from dmeq import age_proportions
    from msinf.age_grid import age_grid_array

    from examples.malaria import analysis

    ages = age_grid_array(dtype=np.float64)
    for name, mu in (
        ("uniform", np.full(ages.shape, analysis.DEATH_RATE)),
        ("rising with age", analysis.DEATH_RATE * (1.0 + np.linspace(0.0, 3.0, ages.size) ** 2)),
    ):
        theirs = np.asarray(age_proportions(ages, mu, dtype=np.float64))
        ours = analysis.age_proportions(ages, mu)
        assert np.allclose(ours, theirs, rtol=1e-12, atol=0.0), name


@needs_msinf
def test_the_benchs_death_rate_is_dmeqs_default(f64):
    """The other copied constant. `prop` is wrong everywhere if this drifts."""
    from dmeq import default_parameters

    from examples.malaria import analysis

    assert analysis.DEATH_RATE == pytest.approx(float(default_parameters()["eta"]), rel=1e-12)


@needs_msinf
def test_an_absorbing_terminal_class_is_refused(bench_env):
    """dmeq raises here too: with no ageing out and no death, it accumulates everyone."""
    import numpy as np

    ages = np.linspace(0.0, 100.0, 20)
    mu = np.full(ages.shape, bench_env.DEATH_RATE)
    mu[-1] = 0.0
    with pytest.raises(ValueError, match="absorbing"):
        bench_env.age_proportions(ages, mu)
