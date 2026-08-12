"""The two models the search starts from, and what the proposer is told.

Two rewritable regions rather than birthdates' one, and the split is forced
rather than chosen: NumPyro sample sites cannot be created inside the `vmap`
over sites, so a candidate declares its parameters in one block and evaluates
curves in another.

    # @discovery:begin parameters   sample statements, once per evaluation
    # @discovery:begin curves       pure jax, called under vmap, once per site

Everything else is scaffolding the proposer may not touch: the study random
effects, the incidence and prevalence likelihoods, the site EIR priors,
`ft_study`, the age grid, the demography and the age-bin averaging. That is what
makes two candidates comparable -- each is a different story about the map from
parameters to age curves and the same story about how a count is drawn given
those curves, so a better held-out likelihood means a better account of malaria
and not a wider noise model fitted to the residuals.

There are two starting points because the point of the run is to compare them,
and they share scaffolding, budget and scores exactly:

* `SCRATCH` -- deliberately poor. Prevalence and incidence that respond to EIR
  and to nothing else: no immunity, no age structure at all. The analogue of
  birthdates' level-and-trend.
* `GRIFFIN` -- the incumbent, for model *correction* rather than discovery. It
  is misspecified; the interesting question is whether a search seeded with it
  beats a search started from nothing, and whether the two converge anywhere
  near each other.

`GRIFFIN` is written out rather than imported from `msinf.curves`, and that is
the duplication `TODO_discovery.md` item 4 is about: a proposer cannot correct a
mechanism it cannot read. The prior *constants* still live in `msinf.priors` --
only the structure is restated -- and `tests/test_malaria.py` fixes a seed and a
parameter vector and asserts the two log-densities agree.
"""

#: what a candidate's source must leave behind: the `msinf.curves.CurveProvider`
#: pair, as two module-level callables
PARAMETERS = "parameters"
CURVES = "curves"

#: the deliberately poor start
SCRATCH = '''\
"""Malaria age curves from EIR alone: no immunity, no age structure."""

import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import sample


def parameters():
    """Sample everything the curves need, once per model evaluation.

    Runs outside the `vmap` over sites, which is the only reason it is a
    separate function: every `numpyro.sample` the candidate wants has to happen
    here. Return whatever `curves` needs -- a dict is conventional, but any
    pytree works.
    """
    # @discovery:begin parameters
    return {
        "prev_max": sample("prev_max", dist.Beta(2.0, 2.0)),
        "prev_halfeir": sample("prev_halfeir", dist.LogNormal(2.3, 1.0)),
        "inc_scale": sample("inc_scale", dist.LogNormal(-1.0, 1.0)),
    }
    # @discovery:end


def curves(params, log_eir, ft, ages, prop, mu):
    """One site's age curves. Pure jax: no sample statements below this line.

    Arguments, all for the one site being evaluated:

      params   -- whatever `parameters()` returned
      log_eir  -- log of the site's annual entomological inoculation rate, scalar
      ft       -- the study's treatment rate, in [0, 1], scalar
      ages     -- lower edge of each model age class, years, shape (165,).
                  Griffin's grid: ~2-week classes through infancy, coarsening to
                  2 years in adulthood. It is not uniform, so a mean over it is
                  not a mean over people.
      prop     -- the fraction of the population in each age class, shape (165,).
                  Fixed demography, supplied as data; the likelihood uses it to
                  average these curves over each observation's age bin.
      mu       -- per-day death rate for each age class, shape (165,). The other
                  half of the demography: it moves an immunity recursion through
                  `r + mu`, not only the age weighting.

    Assign two arrays shaped like `ages`, both as rates per person:

      prev_m -- microscopy-detectable parasite prevalence, in [0, 1]
      inc    -- clinical incidence, episodes per person per year, >= 0

    There is no PCR prevalence: the 2014 table carries no such rows. `prop` and
    `mu` may be ignored by a candidate that has no use for them.
    """
    # @discovery:begin curves
    eir = jnp.exp(log_eir)
    saturation = eir / (eir + params["prev_halfeir"])
    flat = jnp.ones_like(ages)
    prev_m = params["prev_max"] * saturation * flat
    inc = params["inc_scale"] * saturation * flat
    # @discovery:end

    # Fixed, and not the interesting part: msinf averages a curve over an
    # observed age bin as `sum_a w(a) curve(a) / sum_a w(a) prop(a)`, and its
    # incidence likelihood works in person-days. What it has to be handed is
    # therefore a population fraction per day, not a rate per year. Converting
    # here rather than in the block above is what lets a candidate be written in
    # the two quantities it can actually reason about.
    return prev_m * prop, inc * prop / 365.0
'''

#: the incumbent, for correction rather than discovery
GRIFFIN = '''\
"""The Griffin equilibrium model: the incumbent, as it is fitted today.

Acquired immunity in three parts. Exposure to bites builds anti-infection
immunity (IB), which lowers the probability b that a bite infects; infection
builds clinical immunity (IC, plus maternal PM decaying over dm days), which
lowers the probability phi that an infection becomes a clinical episode; and it
builds detection immunity (ID), which lowers the probability q that an infection
is dense enough for microscopy to see. Each is a backward-Euler recursion over
the age grid with its own half-point (IB0/IC0/ID0), shape (kb/kc/kd) and decay
(ub/uc/ud), which is where the age structure comes from: incidence peaks in
early childhood and falls, and it peaks earlier where transmission is higher.

It is misspecified, and it is the thing to beat, not the thing to match.
"""

import jax.numpy as jnp
from dmeq import default_parameters, solve
from msinf.config import resolve_priors, sample_param
from msinf.dtypes import FloatT, asfloat

#: Xi's refit priors, by name. The constants live in `msinf.priors` so that this
#: model and the one msinf fits cannot disagree about them.
PRIORS = resolve_priors("xi")

#: the dmeq parameters the fit samples, in the order the solve applies them
SAMPLED = (
    "b0", "phi0", "phi1", "d1", "PM", "fd0", "ID0", "kd", "ud", "ad0",
    "gd", "IB0", "kb", "ub", "IC0", "kc", "uc", "dm",
)


def parameters():
    """The Griffin parameter block: the dmeq defaults with the sampled ones over."""
    # @discovery:begin parameters
    sampled = {name: sample_param(name, PRIORS[name]) for name in SAMPLED}
    params = {k: asfloat(v) for k, v in default_parameters().items()}
    params.update(sampled)
    # The Griffin parameterization keeps ad0 -- the age at which the detection
    # half-point has fallen halfway -- on a years scale in the prior. The ODE
    # wants days.
    params["ad0"] = params["ad0"] * 365.0
    return params
    # @discovery:end


def curves(params, log_eir, ft, ages, prop, mu):
    """One site's equilibrium solve. See the scratch model for the contract."""
    # @discovery:begin curves
    p = dict(params)
    p["EIR"] = jnp.exp(log_eir)
    p["ft"] = ft
    p["eta"] = mu
    out = solve(p, dtype=FloatT(), age_bins_years=ages)
    # Row 0 is microscopy prevalence, row 1 PCR prevalence (nothing observes
    # it), row 2 clinical incidence -- all as population fractions per
    # person-day, which is dmeq's convention and not this function's. `prop` is
    # bitwise the age distribution dmeq computed internally from `mu`, so
    # dividing it back out is exact and leaves the per-person rates the
    # scaffolding below expects.
    prev_m = out[0, :] / prop
    inc = out[2, :] * 365.0 / prop
    # @discovery:end

    # Fixed, and not the interesting part: msinf averages a curve over an
    # observed age bin as `sum_a w(a) curve(a) / sum_a w(a) prop(a)`, and its
    # incidence likelihood works in person-days. What it has to be handed is
    # therefore a population fraction per day, not a rate per year. Converting
    # here rather than in the block above is what lets a candidate be written in
    # the two quantities it can actually reason about.
    return prev_m * prop, inc * prop / 365.0
'''

#: the starting models by name, for `--start`
STARTS = {"scratch": SCRATCH, "griffin": GRIFFIN}

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
[0, {inc_max:g}] episodes per person-year; and prevalence in 2-10 year olds must
rise with EIR rather than being flat or inverted. A candidate that fails a gate
is reported with its reason and never fitted.
"""
