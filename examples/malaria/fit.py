"""One evaluator: a NUTS fit per split, every score and every plot that comes out.

The harness's unit of work is one evaluator on one model, and its result is a
dictionary -- so identifiability, held-out likelihood, coverage, convergence,
cost and the pictures are keys of one dictionary returned by one call and cannot
disagree about which fits they describe.

One thing differs from `birthdates/fit.py`, and it comes from the child needing
`msinf`: a much longer clock. One fit of this model is tens of minutes rather
than tens of seconds, so `timeout` here is hours where birthdates' is minutes.
It is still a wall clock and not a budget: the budget is
`warmup`/`samples`/`chains`, identical for every candidate, and the timeout only
bounds what a pathological candidate can cost.
"""

import asyncio
import base64
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from discovery.evaluator import Artifact, Diagnostics
from discovery.model import Model

from examples.malaria import data

#: where `python -m examples.malaria` has to be run from
ROOT = Path(__file__).resolve().parents[2]

#: how much of a failed run's stderr to carry back in the error
TAIL = 3000


class FitError(RuntimeError):
    """The child could not fit the model, and this is what it said."""


class GateError(RuntimeError):
    """The candidate failed a pre-fit gate and was never fitted.

    Its own error, and not a `FitError`, because it is not a failure of the fit:
    it is the cheap answer that made the fit unnecessary. It costs one `Result`
    instead of forty minutes, it carries every reason rather than the first, and
    it lands in the report exactly like any other failure -- which is the point.
    The models worth looking at are often the ones that failed.
    """


@dataclass(frozen=True)
class Budget:
    """What every candidate gets, identically.

    A constant budget is what makes the numbers comparable: a model that only
    looks better because it was given more warmup has not been found to be
    better.

    The one place the MCMC budget is set: `__main__` fits with it, and
    `agentic` tells the agents it, so the two cannot disagree. 150 + 150 is the
    screening budget every candidate gets; a candidate worth believing is refitted
    at msinf's own 400 + 400 (`--warmup 400 --samples 400`) before it is called a
    win. `identified` is the KL, in nats, below which a parameter counts as not
    updated by the data.

    Eight chains, jittered starts, and the chains *stacked* rather than pooled.
    This posterior has modes NUTS does not cross: the incumbent's chains split
    between a Dicko Doneguebougou EIR near 50 and one near 320, each with its own
    immunity parameters, at peaks within about two nats of each other, under the
    study split and on the whole table alike; most candidates do the same at one
    site or another. Pooled, a score is a mixture weighted by how many chains
    happened to land in each basin -- the incumbent's held-out elpd is -186 from
    one and -191 from the other, wider than the spread of the whole leaderboard
    -- and a pooled PSIS-LOO falls below every one of its own chains. Stacking
    (`combine_chains`, see `msinf.evaluate`) weights each chain by how well it
    predicts the training rows, so a score depends on which modes were found and
    not on how many chains found each. More chains find more of them; they run
    one thread each, so that costs cores, not wall clock. Every draw is scored
    (`n_score_draws=None`) because each chain needs enough of its own for a
    leave-one-out.
    """

    table: str = str(data.DEFAULT_TABLE)
    eir_priors: str = str(data.DEFAULT_EIR_PRIORS)
    mortality: str | None = data.DEFAULT_MORTALITY
    #: the splits to fit, one fit each. One of them -- see `data.SPLITS`.
    splits: tuple[str, ...] = data.SPLITS
    holdout_seed: int = 42
    n_studies: int = 3
    warmup: int = 150
    samples: int = 150
    chains: int = 8
    max_tree_depth: int = 10
    seed: int = 0
    prior_family: str = "xi"
    #: posterior draws the (ODE-bound) scoring pass runs on -- every one, a few
    #: minutes at most -- and the predictive replicates drawn for the PPC, the
    #: coverage and the randomised PIT
    n_score_draws: int | None = None
    n_replicates: int = 200
    #: how the chains are combined into the scored predictive: one of
    #: `msinf.evaluate.CHAIN_COMBINATIONS`
    combine_chains: str = "stack"
    #: prior draws for the identifiability comparison. Cheap: the likelihood is
    #: blocked, so none of them costs an equilibrium solve.
    prior_draws: int = 2000
    identified: float = 0.05
    max_parameters: int = 200
    #: prior draws the gates probe the curves at, and the site whose fixed
    #: demography they are probed with
    gate_draws: int = 8
    gate_site: int = 0
    inc_max: float = 20.0


#: how a round's scores were computed, written into its `fit.json`: rounds scored
#: differently are not ranked against each other, and `rescore` brings an old
#: round under this one. A `fit.json` without it was scored by pooling the chains
#: and leaving each unit out on its own (`msinf.evaluate.score`).
SCORING = {"combine_chains": Budget.combine_chains, "loo": "conditional"}


def scoring_of(meta: dict) -> dict:
    """How a round's `fit.json` says it was scored."""
    return meta.get("scoring") or {"combine_chains": "pool", "loo": "marginal"}


async def run(source: str, budget: Budget, timeout: float | None = None, samples: dict | None = None) -> dict:
    """Fit one model's source in a child process and read back its report.

    `samples` maps a split to the archive of chains already sampled for it, which
    are then scored again rather than fitted (see `rescore`).
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "examples.malaria.runner",
        cwd=ROOT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    request = json.dumps(asdict(budget) | {"source": source, "archive": samples or {}}).encode()
    try:
        async with asyncio.timeout(timeout):
            out, err = await process.communicate(request)
    except asyncio.CancelledError:
        # Cancelled from outside: the child is ours to clean up, the unwinding
        # is not ours to stop.
        process.kill()
        await process.wait()
        raise
    except TimeoutError as error:
        process.kill()
        await process.wait()
        raise FitError(f"the fit did not finish within {timeout:.0f}s") from error
    if process.returncode != 0:
        raise FitError(f"fit exited {process.returncode}:\n{err.decode(errors='replace')[-TAIL:]}")
    try:
        return json.loads(out)
    except json.JSONDecodeError as error:
        raise FitError(f"fit returned no report:\n{err.decode(errors='replace')[-TAIL:]}") from error


@dataclass
class Posterior(Diagnostics):
    """Everything the fits have to say about one candidate curve family.

    Prefixed by split -- `study.` -- so the key says which fold it came from:

     * `elpd_holdout` -- held-out density with the random effects marginalised
       exactly as the fitted likelihood marginalises them, and `elpd_train` on
       the rows the fit saw. Split into `_incidence` and `_prevalence`, and per
       unit as `_per_cell` and `_per_row`: incidence is scored per *cell*
       (site x age-group), not per row, because `v_jk` is integrated out at the
       cell level and the marginal does not factorise.
     * `elpd_holdout_se` -- that total's own standard error over its units, so a
       total is not read as though it were exact.
     * `pointwise_holdout_incidence`, `pointwise_holdout_prevalence`, and the
       `_train` pair -- the per-unit vectors those totals sum, in the fold's own
       order. Two candidates fitted under the same split line up elementwise,
       which is what makes `estimators.elpd_diff` able to say whether a lead is
       real. This is the only metric here that is a vector rather than a number.
     * `coverage90_holdout_inc`, `coverage90_holdout_prev_m` -- how often the
       held-out count fell inside the 90% predictive interval.
     * `kl.<parameter>` -- the information the data added to each of the
       candidate's own parameters, in nats, on the unconstrained scale;
       `unidentified` counts those it did not move.
     * `max_r_hat`, `min_ess`, `min_ess_tail`, `n_divergent`, `mcmc_seconds`.

    And unprefixed, across every split fitted: `rhat_max`, `ess_tail_min` and
    `divergences` are the worst of them rather than the average, since a
    candidate that converged under one fold and not another has not converged;
    `kl_min` is the least-identified parameter under any; `parameters` is how
    many scalar coordinates the candidate declared.

    The pictures are the posterior predictive check and the randomised-PIT
    calibration plot, for the training and held-out rows of each split.
    """

    budget: Budget = field(default_factory=Budget)
    #: a fit of tens of minutes per split, plus a scoring pass that re-solves the
    #: equilibrium once per retained draw
    timeout: float | None = 7200.0

    async def score(self, model: Model) -> dict:
        report = await run(model.source, self.budget, self.timeout)
        if reasons := report.get("gates"):
            raise GateError("; ".join(reasons))
        # json has no tuples, so the `pointwise_*` vectors arrive as lists and
        # are put back into the shape the harness's `Vector` names.
        metrics = {
            name: tuple(value) if isinstance(value, list) else value for name, value in report["metrics"].items()
        }
        return metrics | dict(artifact(name, payload) for name, payload in report["artifacts"].items())


def artifact(name: str, payload: str) -> tuple[str, Artifact]:
    """One of the runner's base64 payloads, typed by its name, under the key it is reported by.

    A table is keyed without its `.csv`, since the report adds the extension the
    media type implies: `study.fitted_train` lands as `0003-posterior.study.fitted_train.csv`.
    """
    data = base64.b64decode(payload)
    if name.endswith(".samples.npz"):
        return name, Artifact(name, data, "application/npz")
    if name.endswith(".csv"):
        name = name.removesuffix(".csv")
        return name, Artifact(name, data, "text/csv")
    return name, Artifact(name, data)
