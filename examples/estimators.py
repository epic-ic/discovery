"""Diagnostics that are about draws rather than about any one experiment.

Both examples fit a model, get posterior and prior draws back, and have to say
whether the data moved each parameter. That question has nothing to do with
births or with malaria, so it lives here rather than twice in two runners.

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
