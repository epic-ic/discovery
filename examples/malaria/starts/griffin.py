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

Nothing below is written by hand. The mechanism is `dmeq`'s own source and the
parameter block is rendered from `msinf`'s own prior specs, so what you are
reading is what the incumbent is actually fitted with and not a paraphrase that
could have drifted from it.
"""

# This module holds only the imports, because there is nothing about the incumbent
# worth writing by hand: `initial.py` lifts the mechanism out of `dmeq` with
# `inspect.getsource` at both targets, and renders the parameter block out of
# `msinf.priors`. The docstring above becomes the candidate's, so it describes the
# model rather than this file.

# Everything here is unused *in this module* and required by the spliced blocks,
# which are `dmeq`'s source and `initial.py`'s rendering and not this module's --
# see `starts/__init__.py` on why a start is a fragment rather than a runnable
# model. The rendered parameter block writes its priors as literal `sample` calls,
# which is what puts `numpyro` in a preamble that used to import `msinf`.
# ruff: noqa: F401
# @discovery:begin preamble
import numpyro.distributions as dist
from dmeq import backend, default_parameters
from numpyro import sample
# @discovery:end
