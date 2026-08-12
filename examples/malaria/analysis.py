"""What a proposer may look at before it writes a curve family: the training rows.

The rows this run will be scored on are not here, and that is the whole point of
the module. `runner.py` fits one model per split and the report ranks on
`elpd_holdout`; a proposer that could measure the rows behind that number would
be choosing its curve family with the answer in front of it. So `namespace()`
draws the same splits the fit will draw -- same seed, same `n_studies`, through
`msinf.splits`, which is the package that owns the drawing -- and returns what is
left.

The union, if a run ever fits more than one split: a row held out by any fold is
a row behind a number in the report, and so a row the proposer must not see.
Today `data.SPLITS` is the study fold alone, so the union is that one fold's
rows -- roughly 19% of the table at the default seed, leaving the proposer 181
of 223 rows across the full 0-20y range.

**numpy and pandas only, and it is worth some duplication to keep it that way.**
No jax, no numpyro, no dmeq, and no candidate is executed here. Nothing a
proposer wants before it writes a model needs a backend, and importing one costs
gigabytes of address space per worker for a couple of hundred megabytes of
working set -- so the pool stays as cheap as birthdates'. The gates in `gates.py`
do run a candidate's curves, but they run in the fit subprocess where jax already
lives, and they run whether the proposer looked or not: a candidate that ignores
them costs a `GateError`, not a fit.

What that costs is `age_proportions` and one constant, reimplemented below in
numpy because `dmeq` is jax and that recursion is the one thing here that
genuinely needs it. `tests/test_malaria.py` pins both against dmeq, in the same
spirit as the pin on `initial.GRIFFIN` -- a duplicated formula that nothing
compares is a formula that will drift.

Nothing else is copied. Everything that reads a table -- the study rows, the EIR
and `ft` priors, the splits, and the per-site WPP death rates the fit will weight
by -- comes from `msinf`, which is pandas throughout; `msinf.age_grid` and
`msinf.demography` now defer their jax-touching imports into the functions that
need them, so importing a reader does not start a backend. The bench therefore
resolves the *same* demography the fit resolves, rather than showing a uniform
one and hoping nobody notices.

The heavy imports are inside `namespace`, not at module scope: the driver
imports this module for its `DESCRIPTION` and must stay as light as `data` is.
"""

import os

from examples.malaria import data

#: what the proposer is told is in the namespace
DESCRIPTION = """\
It holds the training rows only -- the rows this run will be scored on have been
removed, and no amount of looking will reveal them:

    df                  a DataFrame, one row per observed age band, columns
                          stream    -- 'incidence' or 'prevalence'
                          rate      -- what was observed: for prevalence the
                                       fraction positive by microscopy, for
                                       incidence clinical episodes per
                                       person-year
                          numer     -- positives, or episodes
                          denom     -- people tested, or person-years
                          age0,age1 -- the band's edges in years, [age0, age1)
                          age_mid   -- (age0 + age1) / 2, for convenience only
                          age_bracket -- the band's index within its study, 1-up
                          eir       -- the site's prior point estimate of annual
                                       entomological inoculation rate
                          eir_category -- 1 to 5, the transmission band the EIR
                                       prior's width is taken from
                          ft        -- the study's prior mean treated fraction,
                                       with ft_alpha and ft_beta beside it
                          site_i    -- the canonical site: one physical site,
                                       shared by its incidence and prevalence
                                       rows, which site_index is not
                          study_index, site_index, country_name, site_name
                          case_detection -- 1 active, 2 weekly, 3 and 6 passive
    ages                the model's age grid, 165 class lower edges in years,
                        fine through infancy and coarse in adulthood
    prop                the fraction of the population in each age class, one
                        row per site and one column per age class, each row
                        summing to 1. Index it by a row's site_i -- there are
                        more rows than sites you can see here, because site_i is
                        numbered over the whole table and a held-out study takes
                        its sites with it. Where no mortality table is
                        configured every site has the same exponential age
                        distribution and every row is identical
    np, pd              numpy and pandas

Four things to know before reading a number off this table.

`eir` and `ft` are inputs, not observations. `eir` is the centre of the site's
log-normal prior on transmission and `ft` the mean of the study's prior on the
treated fraction; the fit moves both. Neither is a measurement, and neither is
scored.

A row is not a point on a curve over age. The likelihood averages the model's
curve over the band, weighted by the demography:

    sum_a w(a) curve(a) / sum_a w(a) prop(a)

over `ages`, where `w(a)` is the fraction of age class `a` the band [age0, age1)
covers and `prop` is that site's row. So a band's value is a population-weighted
mean over the band, not the curve at `age_mid`, and because the population falls
steeply with age, a wide band sits nearer its lower edge than its middle.
`ages` and `prop` are here so you can do that arithmetic yourself on any shape
you are considering; the weights and the averaging itself are fixed scaffolding
you do not write.

Raw rates are not comparable across studies or across detection regimes. The
model gives each study random effects on both streams and each site-age
incidence cell its own effect, all integrated out before your curves are scored,
and it fits one scaling factor for weekly active detection (`case_detection` 2)
and another for passive (3 and 6). Structure you find by pooling studies is
partly structure the observational model already absorbs, and it will not be
what earns a better held-out likelihood.

Nothing here observes PCR prevalence. The table carries clinical incidence and
microscopy prevalence, and those are the two things a candidate is scored on.
"""

#: the study table's observation type codes, as `msinf.model` names them
STREAMS = {1: "incidence", 2: "prevalence"}

#: dmeq's default per-day death rate -- `default_parameters()["eta"]`, a mean
#: lifetime of about 21 years. Written out rather than imported because
#: importing dmeq imports jax. It is the rate every run that predates the WPP
#: mortality tables used, and `runner.py` still uses it whenever `--mortality`
#: is unset, which is every run so far.
DEATH_RATE = 1.305e-4


def age_proportions(ages, mu):
    """Equilibrium fraction of the population in each age class.

    `dmeq.age_proportions` in numpy. The recursion is the one in
    malariasimulation's `get_equilibrium_population`: inflow from the class
    below over the outflow of ageing plus death,

        pop[0] = 1 / (r[0] + mu[0])
        pop[i] = pop[i-1] * r[i-1] / (r[i] + mu[i])

    normalised at the end, so the birth rate cancels. `r` is the per-day rate of
    ageing out of a class, `1/width`, and zero in the terminal class, which is
    absorbing and whose death rate is therefore its only outflow.

    Written as a `cumprod` rather than a loop, which is a different association
    of the same products: it agrees with dmeq to about 1e-15 relative at f64 and
    is not bitwise equal. That is the right bar here -- this array is read, and
    never enters a likelihood.
    """
    import numpy as np

    ages = np.asarray(ages, dtype=np.float64)
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), ages.shape)
    if not mu[-1] > 0.0:
        raise ValueError(
            "the terminal age class must have a strictly positive death rate: "
            "it is absorbing, so death is the only way out of it"
        )
    rate = np.append(1.0 / np.diff(ages * 365.0), 0.0)
    pop = np.concatenate([[1.0], np.cumprod(rate[:-1] / (rate[1:] + mu[1:]))]) / (rate[0] + mu[0])
    return pop / pop.sum()


#: the frame and the demography, built once per worker rather than once per
#: snippet. The table is a few thousand rows and rebuilding it every call would
#: be affordable, but the split has to be drawn identically every time and
#: drawing it once is the way to be sure of that. `namespace` hands out copies,
#: which is what makes them read-only.
_BUILT = None


def splits() -> tuple[str, ...]:
    """The splits this run fits, and whose rows must therefore be hidden."""
    named = os.environ.get("MALARIA_SPLITS") or ",".join(data.SPLITS)
    kinds = tuple(kind.strip() for kind in named.split(",") if kind.strip())
    if unknown := set(kinds) - set(data.SPLITS):
        raise ValueError(f"unknown split(s) {sorted(unknown)}; expected some of {list(data.SPLITS)}")
    return kinds


def build() -> tuple:
    """`(df, ages, prop)`: every row no fold of this run holds out, and the demography.

    The masks come from `msinf.splits` rather than from a reimplementation here:
    it is the package that defines which rows can be scored apart from which, it
    refuses a split that straddles an incidence cell, and a second way to draw
    the same fold is a second way to draw it differently. The same goes for the
    mortality table: `msinf.demography` reads it, and only the recursion that
    turns death rates into an age distribution is written out here.
    """
    import numpy as np
    from msinf.age_grid import age_grid_array
    from msinf.data import (
        encode_canonical_sites,
        load_eir_prior_table,
        load_ft_prior_table,
        load_observed_data,
    )
    from msinf.demography import load_site_death_rates, uniform_death_rates
    from msinf.splits import Split, split_mask

    inputs = data.inputs(
        os.environ.get("MALARIA_TABLE") or data.DEFAULT_TABLE,
        os.environ.get("MALARIA_EIR_PRIORS") or data.DEFAULT_EIR_PRIORS,
    )
    frame, _ = encode_canonical_sites(load_observed_data(inputs.table))

    # The union over every fold this run scores. A row held out by either split
    # is a row behind a number in the report.
    held = np.zeros(len(frame), dtype=bool)
    for kind in splits():
        split = Split(
            kind=kind,
            seed=int(os.environ.get("MALARIA_HOLDOUT_SEED", "42")),
            n_studies=int(os.environ.get("MALARIA_N_STUDIES", "3")),
        )
        held |= split_mask(frame, split)
    if held.all():
        raise ValueError(f"splits {splits()} between them hold out every row; nothing is left to look at")

    eir = load_eir_prior_table(frame, inputs.eir_priors)
    ft = load_ft_prior_table(frame, None)
    ft = ft.assign(ft=ft["ft_alpha"] / (ft["ft_alpha"] + ft["ft_beta"]))

    train = (
        frame.loc[~held]
        .merge(eir[["site_index", "eir_estimate", "eir_category"]], on="site_index", how="left")
        .merge(ft[["study_index", "ft", "ft_alpha", "ft_beta"]], on="study_index", how="left")
        .rename(columns={"eir_estimate": "eir"})
    )
    train["stream"] = train["type"].map(STREAMS)
    train["age_mid"] = (train["age0"].astype(float) + train["age1"].astype(float)) / 2.0
    # numer/denom is the fraction positive for prevalence and episodes per
    # person-year for incidence, because the observational denominator of an
    # incidence row is person-years. The two live in one column because every
    # question worth asking of this table is asked within a stream.
    train["rate"] = train["numer"].astype(float) / train["denom"].astype(float)

    columns = [
        "stream", "rate", "numer", "denom", "age0", "age1", "age_mid", "age_bracket",
        "eir", "eir_category", "ft", "ft_alpha", "ft_beta", "site_i",
        "study_index", "site_index", "country_name", "site_name", "case_detection",
    ]
    train = train[[c for c in columns if c in train.columns]].reset_index(drop=True)

    # f64 explicitly: the default asks msinf.dtypes what precision the run is
    # at, and asking imports jax.
    ages = age_grid_array(dtype=np.float64)
    # `(n_site, n_age)` indexed by the canonical `site_i`, exactly like msinf's
    # own `prop_site`, whether or not the rates vary by site. Same shape either
    # way, so a snippet written against one run works against the other.
    n_site = int(frame["site_i"].max()) + 1
    mortality = os.environ.get("MALARIA_MORTALITY")
    if mortality:
        # msinf's reader, not a copy of it: it is pandas throughout, it averages
        # the raw site indices a canonical site aggregates the same way the EIR
        # priors are averaged, and it refuses a table that is off the age grid.
        # Only the recursion it feeds is reimplemented here.
        mu = load_site_death_rates(df=frame, mortality_csv=mortality, n_site=n_site, ages=ages)
    else:
        mu = uniform_death_rates(n_site, int(ages.shape[0]), rate=DEATH_RATE)
    prop = np.stack([age_proportions(ages, row) for row in mu], axis=0)
    return train, ages, prop


def namespace() -> dict:
    """A fresh namespace over the training rows.

    Called once per execution, so the copies are what make the data read-only: a
    snippet may do anything it likes to this frame and the next snippet will not
    know about it.

    The columns are deliberately *not* also bound as bare names, for the reason
    `birthdates/analysis.py` gives at length: short column names are exactly what
    an analyst reaches for as loop variables, and an agent that shadows one gets
    a silently wrong answer rather than a `KeyError`.
    """
    global _BUILT
    import numpy as np
    import pandas as pd

    if _BUILT is None:
        _BUILT = build()
    train, ages, prop = _BUILT
    return {"df": train.copy(), "ages": ages.copy(), "prop": prop.copy(), "np": np, "pd": pd}
