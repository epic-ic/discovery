"""Cheap checks on a candidate's curves, before anything is spent fitting it.

A bad `birthdates` candidate raises or hangs, and the harness notices for free.
A bad candidate here *runs*: it returns plausible-looking nonsense -- negative
states, immunity that diverges, NaN gradients that make NUTS sample garbage
without ever erroring -- and the first sign of it is a fit that cost forty
minutes and produced an `elpd` nobody can interpret.

So the curves are evaluated on their own first, at a handful of prior draws and
a spread of EIRs, and asked four questions:

* is everything finite,
* is prevalence a probability,
* is incidence in a band a person could plausibly experience,
* and does prevalence *rise* with transmission?

The last is the one that catches a candidate which is merely arbitrary rather
than merely wrong: any story about malaria in which more infectious bites do not
mean more infection is not a story about malaria. It is checked on the average
over prior draws rather than draw by draw, because a flexible family will invert
the relationship somewhere in its prior without that being what the model says.

A failure is a reason, not an exception: the caller reports it and moves on, and
the models worth looking at are often the ones that failed.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers

#: EIRs the curves are probed at, infectious bites per person per year. Spans
#: the range the study sites actually cover, low to very high transmission.
EIRS = (0.5, 5.0, 50.0, 300.0)

#: a mid treatment rate; `ft` is study-level and none of the gates are about it
FT = 0.3

#: the age window prevalence is summarised over, years. 2-10y is the
#: conventional one and the one the EIR priors were built against.
WINDOW = (2.0, 10.0)

#: clinical episodes per person per year that no age class may exceed. Generous:
#: the highest observed all-age incidence in these studies is well under 5, and
#: the gate is there to catch curves off by orders of magnitude, not to prefer
#: one plausible level over another.
INC_MAX = 20.0

#: how much prevalence in `WINDOW` must gain across `EIRS` to count as
#: responding, in absolute prevalence
RESPONSE = 0.02

#: prior draws the curves are probed at
DRAWS = 8


def window_mean(curve, ages, prop, window=WINDOW):
    """A curve averaged over an age window, weighted by the demography.

    Not a plain mean over the grid: Griffin's age classes are two weeks wide in
    infancy and two years wide in adulthood, so an unweighted mean over classes
    is a mean over the grid's resolution rather than over people -- and it is
    the weighted version the likelihood scores.
    """
    lo, hi = window
    inside = (ages >= lo) & (ages < hi)
    weight = jnp.where(inside, prop, 0.0)
    total = jnp.sum(weight)
    return jnp.sum(weight * curve) / jnp.where(total > 0, total, 1.0)


#: days per year, for turning msinf's per-person-day incidence back into the
#: per-person-year rate everything here and in the brief is stated in
YEAR = 365.0


def rates(prev_m, inc, prop):
    """dmeq's population fractions per day, back as per-person rates per year.

    The one place the example writes this conversion down other than the
    scaffolding of the model itself. What a curve provider hands msinf is
    `prevalence(a) * prop(a)` and `episodes-per-day(a) * prop(a)` -- because the
    likelihood averages a bin as `sum_a w(a) curve(a) / sum_a w(a) prop(a)` and
    scales incidence by 365 -- and none of the questions below are askable in
    those units.
    """
    return prev_m / prop, inc * YEAR / prop


def probe(parameters, curves, *, ages, prop, mu, ft=FT, eirs=EIRS, draws=DRAWS, seed=0):
    """Evaluate the candidate at `draws` prior draws and every EIR in `eirs`.

    Returns `(prev, inc)` as per-person rates, each shaped
    `(draws, len(eirs), len(ages))`. Prior draws rather than a single median
    because a family can be well behaved at its median and produce NaNs a little
    away from it, which is precisely where warmup will take it.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), draws)
    log_eir = jnp.log(jnp.asarray(eirs, dtype=ages.dtype))

    prev, inc = [], []
    for key in keys:
        params = handlers.seed(parameters, rng_seed=key)()
        pair = jax.vmap(lambda le: curves(params, le, ft, ages, prop, mu))(log_eir)
        prev_rate, inc_rate = rates(pair[0], pair[1], prop)
        prev.append(prev_rate)
        inc.append(inc_rate)
    return np.asarray(jnp.stack(prev)), np.asarray(jnp.stack(inc))


def check(
    parameters,
    curves,
    *,
    ages,
    prop,
    mu,
    ft=FT,
    eirs=EIRS,
    draws=DRAWS,
    seed=0,
    inc_max=INC_MAX,
    response=RESPONSE,
    window=WINDOW,
) -> tuple[str, ...]:
    """Every reason this candidate should not be fitted. Empty means fit it.

    Every gate is reported, not just the first: a candidate that fails three of
    them is a different diagnosis from one that fails one, and the whole point of
    keeping failures in the report is that somebody reads them.
    """
    try:
        prev, inc = probe(
            parameters, curves, ages=ages, prop=prop, mu=mu, ft=ft, eirs=eirs, draws=draws, seed=seed
        )
    except Exception as error:  # noqa: BLE001 -- any failure here is the finding
        return (f"the curves raised on prior draws: {type(error).__name__}: {error}",)

    reasons: list[str] = []
    if prev.shape != inc.shape or prev.shape[-1] != len(ages):
        return (
            f"curves returned {prev.shape} and {inc.shape}; both must be shaped "
            f"like the {len(ages)}-class age grid",
        )

    for name, values in (("prevalence", prev), ("incidence", inc)):
        bad = int(np.sum(~np.isfinite(values)))
        if bad:
            reasons.append(
                f"{name} is not finite at {bad}/{values.size} of the probed "
                f"(prior draw, EIR, age class) points"
            )

    finite_prev = prev[np.isfinite(prev)]
    if finite_prev.size and (finite_prev.min() < 0.0 or finite_prev.max() > 1.0):
        reasons.append(
            f"prevalence leaves [0, 1]: it ranges over "
            f"[{finite_prev.min():.3g}, {finite_prev.max():.3g}]"
        )

    finite_inc = inc[np.isfinite(inc)]
    if finite_inc.size and (finite_inc.min() < 0.0 or finite_inc.max() > inc_max):
        reasons.append(
            f"incidence leaves [0, {inc_max:g}] episodes per person-year: it ranges over "
            f"[{finite_inc.min():.3g}, {finite_inc.max():.3g}]"
        )

    # The response to transmission, averaged over prior draws: one draw of a
    # flexible family may invert it without the model saying anything of the kind.
    ages_j, prop_j = jnp.asarray(ages), jnp.asarray(prop)
    with np.errstate(invalid="ignore"):
        # An all-NaN age class has already been reported above; averaging it
        # here would only add a warning to a finding that is already made.
        over_draws = np.where(
            np.all(np.isnan(prev), axis=0),
            np.nan,
            np.nanmean(np.where(np.isnan(prev), 0.0, prev), axis=0),
        )
    by_eir = np.asarray(
        [
            float(window_mean(jnp.asarray(over_draws[k]), ages_j, prop_j, window))
            for k in range(prev.shape[1])
        ]
    )
    if not np.all(np.isfinite(by_eir)):
        reasons.append("prevalence in the 2-10y window is not finite at every probed EIR")
    elif (gain := by_eir[-1] - by_eir[0]) < response:
        direction = "falls with" if gain < 0 else "is flat in"
        reasons.append(
            f"prevalence in the {window[0]:g}-{window[1]:g}y window {direction} EIR: "
            f"{by_eir[0]:.3g} at EIR {eirs[0]:g} against {by_eir[-1]:.3g} at EIR {eirs[-1]:g}, "
            f"a gain of {gain:+.3g} where at least {response:g} is required"
        )

    return tuple(reasons)
