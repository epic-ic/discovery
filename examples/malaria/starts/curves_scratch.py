"""Malaria age curves from EIR alone: no immunity, no age structure.

Prevalence and incidence saturate in transmission intensity and respond to
nothing else -- not to age, not to treatment, not to the demography, and not to
anything a host has been exposed to before. The equilibrium states, the immunity
recursion and the heterogeneity quadrature are all simply absent: there is no
mechanism here to correct, only a level that moves with EIR.

The analogue of `birthdates`' level-and-trend, and deliberately poor. What a
search finds from here can be read off the distance it travels.
"""

# @discovery:begin preamble
import numpyro.distributions as dist
from dmeq import backend
from numpyro import sample
# @discovery:end


def parameters():
    """Three scalars: a ceiling, a half-saturation EIR, and an incidence scale."""
    # @discovery:begin parameters
    return {
        "prev_max": sample("prev_max", dist.Beta(2.0, 2.0)),
        "prev_halfeir": sample("prev_halfeir", dist.LogNormal(2.3, 1.0)),
        "inc_scale": sample("inc_scale", dist.LogNormal(-1.0, 1.0)),
    }
    # @discovery:end


# @discovery:begin curves
def mechanism(params, log_eir, ft, ages, prop, mu):
    """Flat in age, saturating in EIR."""
    xp = backend().xp
    eir = xp.exp(log_eir)
    saturation = eir / (eir + params["prev_halfeir"])
    flat = xp.ones_like(ages)
    return params["prev_max"] * saturation * flat, params["inc_scale"] * saturation * flat
# @discovery:end
