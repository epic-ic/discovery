"""Immunity that does not accumulate: three constants, no recursion, no memory.

The `immunity` target's analogue of flat-in-age. Every bite infects with the same
probability, every infection is clinical with the same probability, and every
asymptomatic infection is detectable with the same probability -- whatever the
host's age and whatever it has been exposed to before. There is no backward-Euler
recursion, no exposure history and no maternal term, so every age pattern left in
the curves comes from the equilibrium state solve and the demography, both of
which are fixed at this target.

Deliberately poor, and poor in the specific way that matters: Griffin's account
of why malaria incidence peaks in early childhood and peaks earlier where
transmission is higher is entirely a story about accumulated immunity, and this
model has none. What a search recovers from here is the distance between "no
memory" and whatever it invents.
"""

# `Immunity` is deliberately not imported: the template defines it above the block
# with `dmeq`'s own source, so the type a candidate returns is one it can read
# rather than one behind an import. That leaves it undefined *in this file*, which
# is what makes a start a fragment -- see `starts/__init__.py`.
# ruff: noqa: F821
# @discovery:begin preamble
import numpyro.distributions as dist
from dmeq import backend, default_parameters
from numpyro import sample
# @discovery:end


def parameters():
    """dmeq's defaults, with the three constants this immunity model reads over them.

    The defaults are still needed in full: the equilibrium state solve below
    reads recovery rates, treatment parameters and the demography out of the same
    dictionary, and none of that is this candidate's business.
    """
    # @discovery:begin parameters
    params = dict(default_parameters())
    params.update(
        b0=sample("b0", dist.Beta(2.0, 8.0)),
        phi0=sample("phi0", dist.Beta(2.0, 8.0)),
        d1=sample("d1", dist.Beta(5.0, 5.0)),
    )
    return params
    # @discovery:end


# @discovery:begin immunity
def immunity(eps, grid, re, p):
    """Constant susceptibility: `Immunity` with no dependence on exposure or age.

    `eps` is the per-day EIR each age class experiences, already scaled by the
    heterogeneity node and the relative biting rate, shape (n_age,). `grid` is
    the age grid and `re` the per-day rate of leaving an age class; neither is
    read here, because nothing in this model accumulates over age.
    """
    xp = backend().xp
    ones = xp.ones_like(eps)
    b = p["b0"] * ones
    return Immunity(foi=b * eps, phi=p["phi0"] * ones, q=p["d1"] * ones, b=b)
# @discovery:end
