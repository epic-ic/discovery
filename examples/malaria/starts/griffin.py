"""The Griffin equilibrium model: the incumbent, as it is fitted today.

Acquired immunity in three parts. Exposure to bites builds anti-infection
immunity (IB), which lowers the probability b that a bite infects; infection
builds clinical immunity (IC, plus maternal PM decaying over dm days), which
lowers the probability phi that an infection becomes a clinical episode; and it
builds detection immunity (ID), which lowers the probability q that an infection
is dense enough for microscopy to see. Each is a backward-Euler recursion over
the age grid with its own half-point (IB0/IC0/ID0), shape (kb/kc/kd) and decay
(ub/uc/ud), and that recursion is where all of the model's age structure comes
from: incidence peaks in early childhood and falls, and it peaks earlier where
transmission is higher. Downstream of it, each age class is solved for its
equilibrium split across six states -- treated, clinical, prophylactic,
asymptomatic, subpatent and susceptible -- and the two observed quantities are
read off that.

It is misspecified, and it is the thing to beat, not the thing to match.

Everything below the parameter block is `dmeq`'s own source, extracted verbatim
rather than restated, so what you are reading is what the incumbent is actually
fitted with and not a paraphrase that could have drifted from it.
"""

# This module holds only the parameter block and the imports, because there is
# nothing about the incumbent's mechanism worth writing by hand: `initial.py` lifts
# it out of `dmeq` with `inspect.getsource` at both targets. The docstring above
# becomes the candidate's, so it describes the model rather than this file.

# `backend` is unused *here* and required by the spliced mechanism, which is
# `dmeq`'s source and not this module's -- see `starts/__init__.py` on why a start
# is a fragment rather than a runnable model.
# ruff: noqa: F401
# @discovery:begin preamble
from dmeq import backend, default_parameters
from msinf.config import resolve_priors, sample_param
# @discovery:end


def parameters():
    """The Griffin parameter block: the dmeq defaults with the sampled ones over.

    Everything the block needs is inside it, priors and names included, because
    both targets splice this same block and a block that reached for a
    module-level constant would not travel.
    """
    # @discovery:begin parameters
    # Xi's refit priors, by name. The constants live in `msinf.priors` so that
    # this model and the one msinf fits cannot disagree about them.
    priors = resolve_priors("xi")
    # The dmeq parameters the fit samples. Every one of them is read by the
    # immunity model and none by the equilibrium state solve, which is why the two
    # targets sample exactly the same thing and their scores are comparable.
    names = (
        "b0", "phi0", "phi1", "d1", "PM", "fd0", "ID0", "kd", "ud", "ad0",
        "gd", "IB0", "kb", "ub", "IC0", "kc", "uc", "dm",
    )
    sampled = {name: sample_param(name, priors[name]) for name in names}
    params = dict(default_parameters())
    params.update(sampled)
    # The Griffin parameterization keeps ad0 -- the age at which the detection
    # half-point has fallen halfway -- on a years scale in the prior. The ODE
    # wants days.
    params["ad0"] = params["ad0"] * 365.0
    return params
    # @discovery:end
