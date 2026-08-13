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
* does prevalence *rise* with transmission,
* and does the model give the same answer on both of `dmeq`'s array backends?

The fourth is the one that catches a candidate which is merely arbitrary rather
than merely wrong: any story about malaria in which more infectious bites do not
mean more infection is not a story about malaria. It is checked on the average
over prior draws rather than draw by draw, because a flexible family will invert
the relationship somewhere in its prior without that being what the model says.

The fifth is a portability check, and it turns a request into a property. A
candidate is asked to write array operations through `dmeq.backend().xp` rather
than `import jax.numpy as jnp` -- it is the same set of names, and it is what
`dmeq`'s own source does, which is what the `immunity` target's proposer reads.
The reason to insist is that the model then runs without jax: an analysis worker
that wants to look at a curve, a portability probe like this one, or anything
downstream that reuses `dmeq` on numpy. A request in the brief would be followed
sometimes; a gate with an actionable error is followed always. It is affordable
because a numpy solve on the Griffin grid costs about 8ms -- see
`dmeq/tests/test_backend.py` for the agreement the comparison relies on.

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


#: how far the two backends' curves may differ, relative to the jax result.
#: `dmeq`'s own backend test pins its solve at ~1e-15; this is loose enough to
#: allow a different summation order and tight enough that a genuinely different
#: computation fails.
BACKEND_RTOL = 1e-9


def draw_plain(parameters, seed=0):
    """One prior draw of the candidate's parameters, as plain numpy leaves.

    `parameters()` is numpyro and therefore jax, always -- sampling is not the
    portable half of a model. Converting the draw is what lets `curves` be run
    afterwards on a backend that has never seen a jax array.
    """
    import jax

    params = handlers.seed(parameters, rng_seed=jax.random.PRNGKey(seed))()
    return jax.tree.map(lambda leaf: np.asarray(leaf, dtype=np.float64), params)


#: candidate blocks the fixed scaffolding calls, rather than this module calling
#: them. At the `immunity` target `curves` belongs to the template and reaches the
#: candidate's block through `dmeq.solve`, so the only way to see what the
#: candidate wrote is to look it up in the module globals and stand in front of it.
WATCHED = ("immunity",)

#: the advice every portability finding ends with. One string because a proposer
#: acts on the instruction, not on the diagnosis.
ADVICE = (
    "Replace `import jax.numpy as jnp` with `from dmeq import backend` and "
    "`xp = backend().xp`: the function names are the same and the model then runs "
    "on either backend"
)


def all_numpy(value) -> bool:
    """Whether a returned value, or every field of one, is a numpy array.

    A `NamedTuple` of arrays and a bare array are both things a block may return,
    and something that is neither is a shape problem for another gate to report.
    """
    if isinstance(value, np.ndarray):
        return True
    try:
        fields = tuple(value)
    except TypeError:
        return False
    return bool(fields) and all(isinstance(field, np.ndarray) for field in fields)


def portable(
    parameters,
    curves,
    *,
    ages,
    prop,
    mu,
    ft=FT,
    eirs=EIRS,
    seed=0,
    rtol=BACKEND_RTOL,
    namespace=None,
):
    """Reasons the model is not backend-portable. Empty means it is.

    One prior draw and every probed EIR: portability is a property of the code
    rather than of the draw, so a second draw re-asks the same question. No `vmap`
    either -- the numpy backend's is a Python loop, and what is being checked is
    the arithmetic, not the mapping.

    Two findings, not one, because agreement alone does not settle it: `jax.numpy`
    accepts numpy input and returns a jax array, so a candidate that imported
    `jnp` directly agrees with itself to the last bit while never having left jax.
    Nor is the returned type enough on its own -- dmeq's numpy `set_at` assigns
    into a numpy array and silently casts a jax value on the way in, so a leak
    inside an `immunity` block never reaches `curves`' output. Hence `WATCHED`:
    the block is replaced by a wrapper that records what it returned, in the pass
    where the answer has to be numpy.
    """
    import jax
    from dmeq import backend, use_backend

    plain = draw_plain(parameters, seed=seed)
    fixed = tuple(np.asarray(a, dtype=np.float64) for a in (ages, prop, mu))
    log_eir = np.log(np.asarray(eirs, dtype=np.float64))

    def under(name):
        """One backend's answers, on that backend's own array types.

        Converting the inputs and not only reading the outputs, because a mixed
        configuration is one that never occurs: the fit hands jax arrays to the jax
        backend and the bench hands numpy arrays to the numpy one. Handing numpy
        arrays to jax would fail on any model that indexes `prop` by a traced index
        -- which the equilibrium state recursion does, once per age class -- and
        that failure would be the probe's, not the candidate's.
        """
        with use_backend(name):
            xp = backend().xp
            params = jax.tree.map(xp.asarray, plain)
            arrays = tuple(xp.asarray(a) for a in fixed)
            return [curves(params, xp.asarray(le), ft, *arrays) for le in log_eir]

    seen: dict[str, list[bool]] = {}
    restore = {}
    for name in WATCHED:
        block = (namespace or {}).get(name)
        if not callable(block):
            continue
        seen[name], restore[name] = [], block

        def watcher(*args, _block=block, _seen=seen[name], **kwargs):
            out = _block(*args, **kwargs)
            _seen.append(all_numpy(out))
            return out

        namespace[name] = watcher

    running = "jax"
    try:
        on_jax = under("jax")
        # The jax pass is entitled to return jax arrays, so only the numpy pass'
        # records are evidence of anything.
        for records in seen.values():
            records.clear()
        running = "numpy"
        on_numpy = under("numpy")
    except Exception as error:  # noqa: BLE001 -- any failure here is the finding
        # Which backend raised, and not a guess at it: a failure under jax is a
        # different diagnosis from a failure under numpy, and the advice below only
        # fits the second.
        raised = f"{type(error).__name__}: {error}"
        return (f"the model raised when run on dmeq's {running} backend: {raised}. {ADVICE}",)
    finally:
        if restore:
            namespace.update(restore)

    for name, records in seen.items():
        if records and not all(records):
            finding = (
                f"`{name}` returned a jax array from a run on dmeq's numpy backend, so "
                f"it does not use the backend it is given. {ADVICE}"
            )
            return (finding,)

    leaked = [
        kind
        for pair in on_numpy
        for kind, value in zip(("prevalence", "incidence"), pair)
        if not all_numpy(value)
    ]
    if leaked:
        finding = (
            f"`curves` returned {min(leaked)} as a jax array from a run on dmeq's numpy "
            f"backend, so it does not use the backend it is given. {ADVICE}"
        )
        return (finding,)

    for kind, index in (("prevalence", 0), ("incidence", 1)):
        for eir, left, right in zip(eirs, on_jax, on_numpy):
            a, b = np.asarray(left[index], dtype=np.float64), np.asarray(right[index])
            if not np.allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True):
                worst = float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))
                finding = (
                    f"{kind} differs between dmeq's jax and numpy backends at EIR "
                    f"{eir:g}: worst relative difference {worst:.3g}, where {rtol:g} is "
                    f"allowed. The two must be the same computation"
                )
                return (finding,)
    return ()


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
    rtol=BACKEND_RTOL,
    namespace=None,
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

    # Last, and separately: it costs a handful of solves on a slow backend, and a
    # candidate whose numbers are nonsense above has a more useful finding waiting
    # for it than a note about which array library it imported.
    reasons.extend(
        portable(
            parameters, curves, ages=ages, prop=prop, mu=mu, ft=ft, eirs=eirs,
            seed=seed, rtol=rtol, namespace=namespace,
        )
    )

    return tuple(reasons)
