"""Diagnostics that are about draws rather than about any one experiment.

Both examples fit a model, get posterior and prior draws back, and have to say
whether the data moved each parameter. That question has nothing to do with
births or with malaria, so it lives here rather than twice in two runners.

`elpd_diff` is here for the same reason from the other end: it is about a
leaderboard rather than about a fit, and a leaderboard of held-out scores is
what both examples produce.

numpy and scipy only: this is imported by runner processes, which are the ones
that already have both, and it must not drag in `discovery` or `jax`.
"""

import numpy as np
from scipy.spatial import cKDTree


def columns(samples, batch: int):
    """Every scalar coordinate of every site, labelled.

    A site of shape `(k,)` becomes `name[0] .. name[k-1]`, so per-parameter
    metrics stay per-parameter however the candidate chose to vectorise. `batch`
    is how many leading dimensions are draws rather than parameters.
    """
    for name, value in samples.items():
        value = np.asarray(value)
        event = value.shape[batch:]
        if not event:
            yield name, value
            continue
        for index in np.ndindex(*event):
            label = f"{name}[{','.join(str(i) for i in index)}]"
            yield label, value[(slice(None),) * batch + index]


def kl(posterior, prior, k: int = 5) -> float:
    """KL(posterior || prior) in nats for one parameter, from draws alone.

    The k-nearest-neighbour estimator of Perez-Cruz: no density model, no
    assumption that either side is Gaussian, which matters because a posterior
    that has learnt something is usually the shape the prior is not. Zero means
    the data did not move this parameter -- it is not identified -- and the
    estimator is clipped there, since only sampling noise takes it below.
    """
    posterior, prior = np.asarray(posterior, float), np.asarray(prior, float)
    posterior = posterior[np.isfinite(posterior)]
    prior = prior[np.isfinite(prior)]
    n, m = len(posterior), len(prior)
    if n <= k or m < k:
        return float("nan")
    # A parameter that never moved has no k-th neighbour at positive distance.
    if np.ptp(posterior) == 0.0:
        return float("nan")
    here = cKDTree(posterior[:, None])
    there = cKDTree(prior[:, None])
    own = here.query(posterior[:, None], k=k + 1)[0][:, k]
    other = there.query(posterior[:, None], k=k)[0][:, k - 1]
    usable = (own > 0) & (other > 0)
    if not usable.any():
        return float("nan")
    estimate = np.mean(np.log(other[usable] / own[usable])) + np.log(m / (n - 1))
    return float(max(estimate, 0.0))


def elpd_diff(a, b) -> tuple[float, float]:
    """`(elpd_a - elpd_b, se)` from two candidates' per-unit elpd vectors.

    The comparison a leaderboard of held-out scores actually needs. Two totals
    and their individual standard errors cannot answer "is A ahead of B",
    because the errors are strongly correlated: both candidates were scored on
    the same units, and a unit that is hard for one is usually hard for both.
    Differencing first removes that shared difficulty, and the standard error of
    the paired difference is routinely a third of either total's own.

    Vehtari, Gelman and Gabry's `se_diff`: `sqrt(n) * sd` over the per-unit
    differences. It inherits the same normal approximation `elpd` totals do, and
    the same caveat -- with a few dozen units and a candidate that is much worse
    than the other, the approximation is optimistic, so a difference under about
    4 units of its own error is not evidence of anything either way.

    What it does *not* cover is which rows landed in the fold. That is not
    sampling variation in the units and no amount of arithmetic over one fold
    can recover it; it takes more folds.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(
            f"elpd vectors must be the same units in the same order; got {a.shape} and {b.shape}. "
            "Two candidates are only comparable if they were scored under the same split."
        )
    difference = a - b
    n = difference.size
    if n < 2:
        return float(np.sum(difference)), float("nan")
    return float(np.sum(difference)), float(np.sqrt(n) * np.std(difference, ddof=1))
