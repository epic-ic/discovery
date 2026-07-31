"""The model the search starts from.

One rewritable region, `process`, holding the whole generating process: it is
what the proposer is asked to find. Everything after it is fixed scaffolding --
in particular the observational likelihood, the last line of the model. Holding
the observation model still is what makes the scores comparable: every candidate
is a different story about how the rate varies and the same story about how a
count is drawn given the rate, so a better held-out likelihood means a better
account of the process and not a wider noise model fitted to the residuals.

The starting process is deliberately poor -- a level and a linear trend. Births
have a yearly season, a strong weekday effect and holidays; none of that is
here, and finding it is the search.
"""

SOURCE = '''\
"""Daily births as a rate varying over time."""

import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import deterministic, sample


def model(features, births=None):
    """Log-rate of births per day.

    `features` holds equal-length arrays over the days being modelled:

      t     -- years since 1969-01-01, float, running on across the holdout
      doy   -- day of the year, 1.0 to 366.0, float
      dow   -- day of the week, 0 for Monday to 6 for Sunday, int
      month -- 0 for January to 11 for December, int
      year  -- years since 1969, 0 to 19, int

    The int ones index directly: `sample("weekday", dist.Normal(0.0, 0.5)
    .expand([7]).to_event(1))[features["dow"]]` is a weekday effect.

    Write `log_rate`, an array as long as the features. Anything numpyro can
    sample is allowed, but keep the parameter count far below the 5844 training
    days, and give every parameter a prior the data can actually update.
    """
    # @discovery:begin process
    intercept = sample("intercept", dist.Normal(9.2, 1.0))
    trend = sample("trend", dist.Normal(0.0, 0.5))

    log_rate = intercept + trend * features["t"]
    # @discovery:end

    # The observation model is fixed: how a count is drawn given the rate is not
    # what is being searched for, and every candidate is scored through it.
    deterministic("log_rate", log_rate)
    concentration = sample("concentration", dist.Exponential(0.01))
    sample("births", dist.NegativeBinomial2(jnp.exp(log_rate), concentration), obs=births)
'''

#: what the proposer is told about the run, alongside the model itself
BRIEF = """\
Not fitted yet. This is what it will be fitted to and scored on.

The data are daily birth counts in the United States, one row per day.
Training days: {train} ({train_span}). These are the only days you are fitted on.
Held-out days: {holdout} ({holdout_span}), used to score the model and never seen by it.
Births per day average {mean:.0f}.

The fit is NUTS in numpyro, {chains} chains, {warmup} warmup and {samples} post-warmup
draws, the same budget for every candidate. A model is scored on its held-out
predictive likelihood, on whether its 95% predictive intervals cover the held-out
days, on whether every parameter is identified -- the posterior has to differ from
the prior -- and on whether the chains converged. A model with more parameters than
the data can pin down will score worse, not better.
"""
