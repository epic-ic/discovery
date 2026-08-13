"""The models the search starts from, assembled rather than written out.

Two rewritable regions rather than birthdates' one, and the split is forced
rather than chosen: NumPyro sample sites cannot be created inside the `vmap`
over sites, so a candidate declares its parameters in one block and evaluates
the mechanism in another.

    # @discovery:begin parameters   sample statements, once per evaluation
    # @discovery:begin curves       the whole mechanism, at `--target curves`
    # @discovery:begin immunity     the immunity model, at `--target immunity`

Everything else is scaffolding the proposer may not touch: the study random
effects, the incidence and prevalence likelihoods, the site EIR priors,
`ft_study`, the age grid, the demography, the age-bin averaging and the unit
conversion. That is what makes two candidates comparable -- each is a different
story about the map from parameters to age curves and the same story about how a
count is drawn given those curves, so a better held-out likelihood means a better
account of malaria and not a wider noise model fitted to the residuals.

**One model, two apertures.** `--target` does not choose between two models; it
chooses how much of one model the markers open up. Both templates are built from
the same `dmeq` source -- the age grid, the immunity recursion, the equilibrium
state solve, the driver -- and the assembled modules define exactly the same
things. What differs is where the markers sit:

* `curves` -- the immunity model, the state solve and the driver are all inside
  the block. A candidate replaces the equilibrium model entirely.
* `immunity` -- only the immunity model is inside; the state solve and the driver
  sit below it as fixed text. Fixed, but *readable*: what consumes the four values
  a candidate returns is on the page, which is how a proposer learns that `b` is
  never read, that `q` multiplies state A alone, and that `foi` and `phi` set both
  incidence and the states it is read off.

`test_the_target_is_an_aperture_over_one_model` pins that. The incumbent also
samples the same 18 parameters at either target -- every one is read by the
immunity model and none by the state solve -- so `griffin` at the two targets is
the same fitted model, and the two runs' scores are directly comparable.

**Two starting points per target.** Scaffolding, budget and scores are identical
between them; only the initial contents of the block differ, which is what a
"start" is:

* `scratch` -- deliberately poor. Curves that respond to EIR and nothing else, or
  immunity that does not accumulate. The analogue of birthdates'
  level-and-trend.
* `griffin` -- the incumbent, for model *correction* rather than discovery. It is
  misspecified; the interesting question is whether a search seeded with it beats
  a search started from nothing, and whether the two converge anywhere near each
  other.

Nothing here is a string literal holding candidate source, and nothing about the
incumbent is transliterated. Each start is a real module under `starts/` whose
marked regions are lifted out with `discovery.model`'s own block parser, and every
piece of the mechanism is `inspect.getsource` from `dmeq` -- so what a proposer
reads is byte for byte what the fit runs and cannot drift from it. The incumbent's
parameter block is the same idea one step removed: it is rendered by
`griffin_parameters` from `msinf`'s own `PriorSpec`s rather than copied, because a
proposer that cannot see a prior's family and scale cannot judge what to keep.
The one real exception is `DRIVER`, 25 hand-written lines standing in for
`dmeq._solve`, and `test_the_incumbent_candidate_agrees_with_msinfs_own_model` is
what stops it drifting -- it pins the rendered priors too. All of this is cheap
only because `import dmeq` imports no array library and `getsource` executes
nothing.
"""

from inspect import getsource
from textwrap import dedent, indent

from msinf.config import resolve_priors

from discovery.model import Model

from examples.malaria.starts import (
    curves_scratch,
    documentation,
    griffin,
    immunity_scratch,
    regions,
)

#: what a candidate's source must leave behind: the `msinf.curves.CurveProvider`
#: pair, as two module-level callables
PARAMETERS = "parameters"
CURVES = "curves"

#: the block a candidate rewrites at each target. `parameters` is rewritable at
#: both; this is the other one.
MECHANISM = {"curves": "curves", "immunity": "immunity"}


def undocumented(source: str) -> str:
    """`source` with its leading definition's docstring removed.

    Extracted source goes inside a candidate's own docstring, and a nested `\"\"\"`
    would end that docstring early -- producing a module that still parses and
    means something else entirely, which is the worst of the available failures.
    So the docstring is dropped rather than escaped, and what is left is the field
    declarations and their comments, which is the part being quoted anyway.

    Located with `ast` rather than by matching quotes, so a docstring that grows to
    several lines is still removed whole, and the result is asserted clean: if
    triple quotes ever survive this, failing at import beats shipping it.
    """
    import ast

    lines = source.splitlines()
    body = ast.parse(dedent(source)).body[0].body
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        del lines[first.lineno - 1 : first.end_lineno]
    out = "\n".join(lines).rstrip()
    if '"""' in out or "'''" in out:
        raise ValueError(f"triple quotes survived docstring removal:\n{out}")
    return out


def backend_contract(prefix: str = "") -> str:
    """What `dmeq.backend()` hands back, in dmeq's own words.

    `Backend`'s source rather than a restatement of it, for the same reason the
    incumbent's immunity block is: a candidate that is told six primitives exist
    and given five of them writes `jax.lax.scan`, and a list of names maintained
    here would go stale the first time dmeq grew a seventh.

    Telling a proposer only about `xp` is what the first version of this did, and
    it is the wrong half to tell: `xp` covers the arithmetic, which is the part
    that needs no instruction because the names are shared. What needs saying is
    the five operations the array API deliberately does not cover -- and one of
    them, a recursion over the age grid, is what an immunity model *is*.
    """
    # `dmeq._backend` and not `dmeq`: `backend()` is exported and the type it
    # returns is not, so there is no public name for the contract it satisfies.
    # `type(backend())` would be public and would resolve a backend -- an import
    # of jax -- to read a class. dmeq should export `Backend`; until it does, this.
    from dmeq._backend import Backend

    fields = undocumented(getsource(Backend))
    body = f"""\
Array operations go through `dmeq.backend()` rather than `import jax.numpy as jnp`.
It hands back the six things that differ between the two backends:

{indent(fields, "    ")}

`xp` needs no further explanation: `exp`, `append`, `arange`, `where` and the rest
are the same names in numpy and in `jax.numpy`, which is why the arithmetic is
portable for free. The other five are the operations the array API does not cover,
and they are the ones worth knowing about here:

  a recursion over age classes -- which is what an immunity model is -- goes
  through `backend().scan` (jax's `lax.scan` contract, `(f, init, xs) ->
  (carry, ys)`) or `backend().fori_loop`. Not through `jax.lax`, which fails the
  portability gate, and not by unrolling a Python loop over all 165 classes,
  which passes the gate and then costs a fit: it traces into 165 copies of the
  body and the compile alone can outrun the timeout.

  functional assignment -- `x[i] = v` on an array you do not own -- goes through
  `backend().set_at(x, i, v)`.

`calculate_immunity` is the backward-Euler recursion Griffin's model is built from,
already written against this interface and readable above the block you are
rewriting. A candidate that wants that recursion under a different exposure or
different constants should call it rather than restate it; a candidate whose story
about immunity is a different recursion should write its own, through
`backend().scan`, and can read that one as the worked example."""
    return indent(body, prefix) if prefix else body



def extract(*objects) -> str:
    """dmeq's own source for each object, in order, verbatim.

    Docstrings and comments included, and unlike `undocumented` they are kept: this
    text is spliced into the module *body*, where a triple quote is not a hazard,
    and dmeq's docstrings are the paragraphs a proposer should read.
    """
    return "\n\n".join(getsource(obj).rstrip() for obj in objects)


def dmeq_blocks() -> dict[str, str]:
    """The equilibrium model in pieces, split by whether a candidate owns it.

    Every piece is `getsource` and therefore cannot drift from what the incumbent
    is fitted with. Which pieces sit inside the markers is the *only* thing
    `--target` changes -- the target is an aperture over one shared model, not two
    different models -- so the split lives here rather than in either template.

    `grid` and `immunity_blocks` are fixed at both targets, for two different
    reasons. The age grid is a deterministic function of `ages`, which is data: a
    candidate that changed class widths or midpoints would desynchronise from the
    `weighted_bin_mean_batch` weights the likelihood averages its bins with, so it
    is scaffolding in the same way `prop` is. The immunity recursion is fixed
    because it is an *offer*: read it, call it, or write your own -- but the
    incumbent's cannot be edited out from under the comparison.

    `dmeq.age_proportions` is deliberately absent. `curves` is handed `prop`, so
    the states are built against the same demography the likelihood weights by,
    and `pos_M / prop` divides by the very array the states summed to. dmeq
    recomputes it internally from `p['eta']` and the two are bitwise equal today;
    taking the argument makes that an identity rather than a coincidence.
    """
    from dmeq.dmeq import (
        AgeGrid,
        Immunity,
        _compute_state,
        _next_immunity,
        _next_state,
        _non_het_prev,
        _resolve_dtype,
        age_grid,
        calculate_immunity,
        griffin_immunity,
    )

    return {
        # fixed at both targets: the geometry of the fixed age grid
        "grid": extract(AgeGrid, _resolve_dtype, age_grid),
        # fixed at both targets: the type the state solve reads, and the recursion
        # Griffin's immunity is built from
        "immunity_blocks": extract(Immunity, _next_immunity, calculate_immunity),
        # searchable at `curves`, fixed at `immunity`: what consumes foi/phi/q/b
        "states": extract(_compute_state, _next_state, _non_het_prev),
        # the incumbent's immunity model, searchable at both targets
        "immunity": extract(griffin_immunity),
    }


#: `Array` and `NamedTuple`, which the extracted `AgeGrid` and `Immunity` need in
#: scope: a NamedTuple's field annotations are evaluated at class creation. dmeq
#: spells it `Any` on purpose -- naming a backend's array type would import that
#: backend -- and the reason is worth carrying across with it.
TYPES = '''\
from typing import Any, NamedTuple

#: An array of whichever backend is in force. Spelled as an alias rather than as
#: `jnp.ndarray` because naming a backend's array type in an annotation would
#: import that backend at class creation, and this module imports neither.
Array = Any'''

#: the heterogeneity quadrature. Numerical constants rather than a claim about
#: malaria, so they are fixed at both targets and written here rather than
#: extracted: dmeq builds them inline inside `_solve`, which is the one function
#: not lifted verbatim. A wrong digit fails the log-density pin.
QUADRATURE = '''\
#: 10-point Gauss-Hermite quadrature over the log-normal heterogeneity in biting
#: rate. `s2` is its variance; `zeta` below is the per-node EIR multiplier.
GH_NODES = (
    -4.8594628, -3.5818235, -2.4843258, -1.4659891, -0.4849357,
    0.4849357, 1.4659891, 2.4843258, 3.5818235, 4.8594628,
)
GH_WEIGHTS = (
    4.310653e-06, 7.580709e-04, 1.911158e-02, 1.354837e-01, 3.446423e-01,
    3.446423e-01, 1.354837e-01, 1.911158e-02, 7.580709e-04, 4.310653e-06,
)'''

#: Griffin's driver: everything `dmeq._solve` does other than build the age grid
#: and the demography. The one piece of the model not lifted verbatim, because
#: lifting it would drag `age_proportions` back into the search space -- so it is
#: 25 hand-written lines against 142 extracted ones, and
#: `test_the_incumbent_candidate_agrees_with_msinfs_own_model` is what stops it
#: drifting from `dmeq.solve`.
DRIVER = '''\
def mechanism(params, log_eir, ft, ages, prop, mu):
    """One site's equilibrium, averaged over the heterogeneity quadrature."""
    be = backend()
    xp = be.xp
    dtype = prop.dtype

    p = dict(params)
    p["EIR"] = xp.exp(log_eir)
    p["ft"] = ft
    grid = age_grid(ages, dtype=dtype)

    # rate of leaving an age class: ageing plus death. The demography is data, so
    # `mu` is used directly rather than read back out of `p`.
    re = grid.ageing + xp.asarray(mu, dtype=dtype)

    # relative biting rate: children are bitten less than adults
    psi = 1. - p["rho"] * xp.exp(-grid.midpoints / p["a0"])

    # EIR multiplier at each quadrature node
    zeta = xp.exp(-p["s2"] * .5 + xp.sqrt(p["s2"]) * xp.asarray(GH_NODES, dtype=dtype))

    rows = be.vmap(
        lambda zeta_i: _non_het_prev(grid, prop, re, psi, p, zeta_i, immunity, dtype)
    )(zeta)
    weights = xp.expand_dims(xp.asarray(GH_WEIGHTS, dtype=dtype), [1, 2])
    averaged = xp.sum(rows * weights, axis=0)

    # Rows 0 and 2 are microscopy prevalence and clinical incidence as population
    # fractions per person-day. `prop` is the array the six states of each age
    # class sum to, so dividing by it is exact and gives per-person rates.
    return averaged[0] / prop, averaged[2] * 365. / prop'''


#: the shared docstring of `parameters()`. One text for both targets: what the
#: function is for and why it is separate does not depend on what is downstream
#: of it.
PARAMETERS_DOC = '''\
    """Sample everything the mechanism needs, once per model evaluation.

    Runs outside the `vmap` over sites, which is the only reason it is a separate
    function: every `numpyro.sample` the candidate wants has to happen here.
    Return whatever the block below needs -- a dict is conventional, but any
    pytree works.
    """'''

#: the fixed `curves`, identical at both targets: the unit conversion of
#: `TODO_discovery.md` section 0 and nothing else. It is outside every marker at
#: both targets, which is the point -- a candidate writes in the two quantities it
#: can reason about and cannot get the population-fraction convention wrong.
CURVES_WRAPPER = '''\
def curves(params, log_eir, ft, ages, prop, mu):
    """One site's age curves, in the units msinf's likelihood is handed.

    Fixed at both targets. See the module docstring for what `mechanism` is given
    and what it must return.
    """
    prev_m, inc = mechanism(params, log_eir, ft, ages, prop, mu)

    # msinf averages a curve over an observed age bin as
    # `sum_a w(a) curve(a) / sum_a w(a) prop(a)`, and its incidence likelihood
    # works in person-days. What it has to be handed is therefore a population
    # fraction per day, not a rate per year. Converting here rather than in the
    # block above is what lets a candidate be written in the two quantities it can
    # actually reason about.
    return prev_m * prop, inc * prop / 365.0'''

#: what the arguments are, at either target. The `curves` signature is the same
#: one `mechanism` is handed, so this is shared.
ARGUMENTS = """\
      params   -- whatever `parameters()` returned
      log_eir  -- log of the site's annual entomological inoculation rate, scalar
      ft       -- the study's treatment rate, in [0, 1], scalar
      ages     -- lower edge of each model age class, years, shape (165,).
                  Griffin's grid: ~2-week classes through infancy, coarsening to
                  2 years in adulthood. It is not uniform, so a mean over it is
                  not a mean over people. `age_grid(ages)` below is its geometry.
      prop     -- the fraction of the population in each age class, shape (165,).
                  Fixed demography, supplied as data. The likelihood averages over
                  age bins with it, and the equilibrium states of each class sum to
                  it, so it is the same array in both roles and neither is yours.
      mu       -- per-day death rate for each age class, shape (165,). The other
                  half of the demography: it moves a recursion over age through
                  `ageing + mu`, not only the age weighting."""

#: the `curves` target's contract: the whole equilibrium model is on the table.
CONTRACT_CURVES = f"""\
## What this target asks for

Rewrite the block, which is the whole equilibrium model: how immunity accumulates
over age, how the states of each age class are solved, and how the two observed
quantities are read off them. What is *not* yours is above it -- the age grid, the
demography, the quadrature constants and the offered immunity recursion -- and the
unit conversion below it.

    mechanism(params, log_eir, ft, ages, prop, mu) -> (prev_m, inc)

{ARGUMENTS}

Return two arrays shaped like `ages`, both as rates per person:

      prev_m -- microscopy-detectable parasite prevalence, in [0, 1]
      inc    -- clinical incidence, episodes per person per year, >= 0

There is no PCR prevalence: the 2014 table carries no such rows. `prop` and `mu`
may be ignored by a candidate that has no use for them, though a story about age
that ignores mortality is a story about a population that never dies."""

#: the `immunity` target's contract: one block, and the machinery that consumes it
#: is on the page.
CONTRACT_IMMUNITY = f"""\
## What this target asks for

Rewrite `immunity` and nothing else. The equilibrium state recursion, the
heterogeneity quadrature, the demography and the unit convention are all fixed and
shared by every candidate -- and they are below the block rather than behind an
import, so what happens to the four values you return can be read rather than
guessed.

    immunity(eps, grid, re, p) -> Immunity(foi, phi, q, b)

      eps   -- the per-day entomological inoculation rate each age class
               experiences, already scaled by the heterogeneity quadrature node
               and by the age-dependent relative biting rate. Shape (n_age,).
      grid  -- the age grid, as the `AgeGrid` defined below.
      re    -- the per-day rate of leaving an age class by ageing *or* death,
               shape (n_age,). A recursion over age that ignores this is a
               recursion over a population that never dies.
      p     -- whatever `parameters()` returned.

Return the four fields the state solve reads, each shaped like `eps`, and no
others -- the immunity *levels* are Griffin's parameterization rather than the
solve's requirement, and a candidate with its own has nothing to put in them:

      foi   -- force of infection, per day
      phi   -- probability an infection becomes a clinical episode, in [0, 1]
      q     -- probability an asymptomatic infection is dense enough to be seen
               by microscopy, in [0, 1]
      b     -- probability an infectious bite infects, in [0, 1]

Read `_non_het_prev` below before deciding what those four should be. Three things
it settles that the list above does not: `b` is stacked into an output row and
never read by the solve, so structure spent on it earns nothing; `q` multiplies
state A alone, so microscopy prevalence is `T + D + A*q` rather than a
detectability applied to everyone infected; and `foi` and `phi` set incidence
through `(S + U + A) * foi * phi` while *also* setting the states they multiply,
through `betaA` and `aT`/`aD`.

The whole model runs under `curves`, which is handed the same arguments:

{ARGUMENTS}"""

CONTRACTS = {"curves": CONTRACT_CURVES, "immunity": CONTRACT_IMMUNITY}

#: one module skeleton for both targets. Sentinels rather than `str.format`
#: throughout, because every piece spliced into it is Python and full of braces.
SKELETON = """\
\"\"\"@@DOC@@

@@CONTRACT@@

## How to write it

@@BACKEND@@

All of it is checked: a candidate that runs on one backend only is rejected by a
gate before it is fitted.
\"\"\"

@@PREAMBLE@@

@@TYPES@@


# ---------------------------------------------------------------------------
# Fixed: the geometry of the age grid, which is a function of `ages` and
# therefore of data. Changing it would desynchronise from the weights msinf
# averages an observed age bin with.
# ---------------------------------------------------------------------------

@@GRID@@


# ---------------------------------------------------------------------------
# Fixed: the type the equilibrium solve reads, and the backward-Euler recursion
# Griffin's immunity is built from. An offer, not a requirement -- call it,
# ignore it, or write your own recursion through `backend().scan`.
# ---------------------------------------------------------------------------

@@IMMUNITY_BLOCKS@@

@@QUADRATURE@@


def parameters():
@@PARAMETERS_DOC@@
    # @discovery:begin parameters
    return {}
    # @discovery:end


@@MECHANISM@@


@@CURVES_WRAPPER@@
"""

#: what sits between `parameters` and `curves`, per target. The markers move and
#: nothing else does: at `curves` they open before the immunity model and close
#: after the driver, at `immunity` they close again immediately and the states and
#: the driver become fixed text below.
MECHANISM_SECTIONS = {
    "curves": """\
# @discovery:begin curves
def mechanism(params, log_eir, ft, ages, prop, mu):
    \"\"\"Replaced by the start, and then by a proposal.\"\"\"
# @discovery:end""",
    "immunity": """\
# @discovery:begin immunity
def immunity(eps, grid, re, p):
    \"\"\"Replaced by the start, and then by a proposal.\"\"\"
# @discovery:end


# ---------------------------------------------------------------------------
# Fixed at this target, and here to be read: this is what consumes the four
# values `immunity` returns. At `--target curves` this same code is inside the
# block above and is yours to change.
# ---------------------------------------------------------------------------

@@STATES@@


@@DRIVER@@""",
}


def template(target: str) -> str:
    """The module every candidate at `target` is a splice into.

    Built once at import. Everything in it other than the two marked regions is
    fixed, and all of it except the driver, the quadrature constants and the type
    alias is `getsource` from dmeq.
    """
    blocks = dmeq_blocks()
    mechanism = (
        MECHANISM_SECTIONS[target]
        .replace("@@STATES@@", blocks["states"])
        .replace("@@DRIVER@@", DRIVER)
    )
    return (
        SKELETON
        .replace("@@CONTRACT@@", CONTRACTS[target])
        .replace("@@BACKEND@@", backend_contract())
        .replace("@@TYPES@@", TYPES)
        .replace("@@GRID@@", blocks["grid"])
        .replace("@@IMMUNITY_BLOCKS@@", blocks["immunity_blocks"])
        .replace("@@QUADRATURE@@", QUADRATURE)
        .replace("@@PARAMETERS_DOC@@", PARAMETERS_DOC)
        .replace("@@MECHANISM@@", mechanism)
        .replace("@@CURVES_WRAPPER@@", CURVES_WRAPPER)
    )


TEMPLATES = {target: template(target) for target in ("curves", "immunity")}


def griffin_mechanism(target: str) -> str:
    """The incumbent's block at `target`, assembled out of dmeq's own source.

    At `immunity` it is `griffin_immunity` and the alias the fixed driver reaches
    it by -- dmeq's function keeps its own name so the extracted text stays byte
    for byte dmeq's, and a test asserts that. At `curves` the same text is followed
    by the state solve and the driver, which are fixed one target down.

    Nothing here is transliterated. `getsource` executes nothing and `import dmeq`
    imports no array library, so the driver can introspect the incumbent without
    starting a backend.
    """
    blocks = dmeq_blocks()
    alias = "#: the name the fixed driver hands to `_non_het_prev`\nimmunity = griffin_immunity"
    parts = [blocks["immunity"], alias]
    if target == "curves":
        parts += [blocks["states"], DRIVER]
    return "\n\n".join(parts)


#: the incumbent's sampled parameters, grouped by which of the three
#: probabilities they move and in the order `griffin_immunity` reaches them. The
#: grouping is the only editorial thing in the rendered block, and it is read off
#: the extracted mechanism sitting a few hundred lines below it.
GRIFFIN_PARAM_GROUPS = (
    ("b, the probability an infectious bite infects, through IB", ("b0", "IB0", "kb", "ub")),
    (
        "phi, the probability an infection becomes a clinical episode, through IC",
        ("phi0", "phi1", "IC0", "kc", "uc", "PM", "dm"),
    ),
    (
        "q, the probability an asymptomatic infection is seen by microscopy, through ID",
        ("d1", "ID0", "kd", "ud", "fd0", "ad0", "gd"),
    ),
)

#: the two parameters whose prior is not on the scale the ODE reads them at. Both
#: are unguessable from the number alone, and one of them is why the block ends
#: with a multiplication.
GRIFFIN_PARAM_UNITS = {"ad0": "years", "dm": "days"}

#: `msinf.config.sample_param`'s dispatch, as the literal call it makes. Rendering
#: the call rather than making it is the whole point: a proposer that cannot see
#: the family and its arguments cannot judge what to keep, and cannot pick a prior
#: for anything it adds.
PRIOR_CALLS = {"beta": "dist.Beta", "lognormal": "dist.LogNormal"}


def prior_line(spec) -> tuple[str, str]:
    """One sampled parameter as code and as the comment that says where it lives.

    Split in two so the caller can align the comments into a column; a prior is
    two numbers whose meaning is a support and a scale, and a column of medians is
    how that gets read at a glance rather than reconstructed.
    """
    call = PRIOR_CALLS.get(spec.family)
    if call is None:
        raise ValueError(f"unsupported family for {spec.name}: {spec.family}")

    def g(value: float) -> str:
        return f"{value:.3g}"

    code = f'{spec.name}=sample("{spec.name}", {call}({spec.param1!r}, {spec.param2!r})),'
    unit = GRIFFIN_PARAM_UNITS.get(spec.name)
    comment = f"# median {g(spec.median)}, 95% ({g(spec.p2_5)}, {g(spec.p97_5)})"
    return code, comment + (f" -- {unit}" if unit else "")


def griffin_parameters() -> str:
    """The incumbent's parameter block, rendered from `msinf`'s own prior specs.

    Written out rather than looked up. `resolve_priors("xi")` and `sample_param`
    keep this model and the one msinf fits from disagreeing, but they do it by
    putting the priors behind two calls whose source is not on the page -- so a
    proposer reads eighteen names and learns nothing about the space they live in:
    not which are probabilities and which are positive reals, not that `IB0` is
    O(20) while `kb` is O(1), not that `dm` is in days, and not how to declare a
    parameter of its own. That is not a search problem, it is a legibility one,
    and `scratch` does not have it: its three priors are literals.

    So the numbers come out of `msinf.priors` and go onto the page as the calls
    `sample_param` would have made. The single source of truth is unchanged --
    this is generated from it, in the same way the mechanism is `getsource` from
    `dmeq` -- and `test_the_incumbent_candidate_declares_the_same_parameters` is
    what holds the rendering to it.
    """
    priors = resolve_priors("xi")
    rendered = [
        (heading, [prior_line(priors[name]) for name in names])
        for heading, names in GRIFFIN_PARAM_GROUPS
    ]
    width = max(len(code) for _, lines in rendered for code, _ in lines)

    body = []
    for heading, lines in rendered:
        body.append(f"        # {heading}")
        body += [f"        {code:<{width}}  {comment}" for code, comment in lines]
    return PARAMETERS_BLOCK.replace("@@PRIORS@@", "\n".join(body))


#: the incumbent's parameter block with its priors left out, because they are
#: generated. Everything around them is fixed prose about what the block is for.
PARAMETERS_BLOCK = '''\
    # Xi's refit priors, written out. The numbers are generated from the specs in
    # `msinf.priors`, so this model and the one msinf fits cannot disagree about
    # them; they are on the page rather than behind a lookup so that what the
    # incumbent samples can be read, and so that a candidate adding a parameter of
    # its own has the idiom in front of it. Each trailing comment is the prior's
    # median and its 95% interval.
    #
    # Every one of these is read by the immunity model and none by the equilibrium
    # state solve, which is why the two targets sample exactly the same thing and
    # their scores are comparable. Nothing requires a candidate to keep them: they
    # are the incumbent's parameterization, not the solve's.
    params = dict(default_parameters())
    params.update(
@@PRIORS@@
    )
    # The Griffin parameterization keeps ad0 -- the age at which the detection
    # half-point has fallen halfway -- on a years scale in the prior. The ODE
    # wants days.
    params["ad0"] = params["ad0"] * 365.0
    return params'''


def assemble(
    target: str, module, mechanism: str | None = None, parameters: str | None = None
) -> str:
    """A start module spliced into its target's template.

    `mechanism` and `parameters` override the module's own blocks, which is how
    the incumbent comes out of `dmeq` and `msinf` rather than out of `starts/` --
    `starts/griffin.py` carries the imports and nothing else, because there is
    nothing about the incumbent worth writing by hand and a hand copy could drift
    from what it is actually fitted with.
    """
    blocks = regions(module)
    if mechanism is not None:
        blocks[MECHANISM[target]] = mechanism
    if parameters is not None:
        blocks[PARAMETERS] = parameters

    missing = [name for name in (PARAMETERS, MECHANISM[target]) if name not in blocks]
    if missing:
        raise ValueError(f"{module.__name__} has no {' or '.join(missing)} region for target {target!r}")

    filled = (
        TEMPLATES[target]
        .replace("@@DOC@@", documentation(module))
        .replace("@@PREAMBLE@@", blocks["preamble"])
    )
    return Model(filled).substitute(
        {PARAMETERS: blocks[PARAMETERS], MECHANISM[target]: blocks[MECHANISM[target]]}
    ).source


#: the search targets: how much of one shared model the markers open up
TARGETS = ("curves", "immunity")

#: the starting points, per target
START_NAMES = ("scratch", "griffin")

#: every starting model, keyed `(target, start)`. Built at import: assembling one
#: is a `getsource` and a splice, and a start that does not splice should fail
#: here rather than an hour into a run.
STARTS: dict[tuple[str, str], str] = {
    ("curves", "scratch"): assemble("curves", curves_scratch),
    ("curves", "griffin"): assemble(
        "curves", griffin, griffin_mechanism("curves"), griffin_parameters()
    ),
    ("immunity", "scratch"): assemble("immunity", immunity_scratch),
    ("immunity", "griffin"): assemble(
        "immunity", griffin, griffin_mechanism("immunity"), griffin_parameters()
    ),
}

#: what the proposer is told about the run, alongside the model itself
BRIEF = """\
Not fitted yet. This is what it will be fitted to and scored on.

The data are age-stratified malaria observations from African field studies:
{table}. Each row is a count in one age band at one site -- clinical incidence
(episodes over person-time) or microscopy prevalence (positive slides over
slides examined). Sites differ enormously in transmission intensity, and the
site's annual EIR is a latent parameter with an informative prior, not something
you are given.

Held-out scoring, one split:

  study     -- {study} whole studies are dropped from the fit, and your model is
               scored on their rows. Know what this is blunt about: a held-out
               study's EIR sits on its prior rather than being informed by its
               own data, so a good part of the number is how your curves behave
               when transmission intensity is only known to within a wide prior.
               That is shared by every candidate. What separates candidates is
               age structure, and it is the smaller part of this number -- so a
               small lead over another candidate is not a real one.

The fit is NUTS in numpyro at f64, {chains} chains, {warmup} warmup and {samples}
post-warmup draws, the same budget for every candidate. One fit takes roughly
{minutes:.0f} minutes, so a candidate that is much more expensive to solve will
be killed rather than scored.

Scored on held-out predictive likelihood with the study random effects
integrated out, on whether the predictive intervals cover the held-out counts,
on whether every parameter is identified -- the posterior has to differ from the
prior -- and on whether the chains converged. More parameters than the data can
pin down scores worse, not better.

Before any of that, a candidate is run through cheap gates on the curves alone:
they must be finite; prevalence must lie in [0, 1]; incidence must lie in
[0, {inc_max:g}] episodes per person-year; prevalence in 2-10 year olds must
rise with EIR rather than being flat or inverted; and the model must give the
same answer on both of `dmeq`'s array backends, which means going through
`dmeq.backend()` and not through `jax.numpy` or `jax.lax` -- the model source you
are given documents the whole of that interface, including the primitive a
recursion over age classes needs. A candidate that fails a gate is reported with
its reason and never fitted.
"""
