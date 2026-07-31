"""One evaluator: one NUTS fit, every score and both plots that come out of it.

The harness's unit of work is one evaluator on one model, and its result is a
dictionary. That is the whole reason nothing needs sharing here: identifiability,
likelihood, coverage, R-hat, tail-ESS, the cost of the fit and the two pictures
are all keys of one dictionary returned by one call, so they cannot disagree
about which fit they describe.

The fit itself happens in a child process. `discovery` has no sandbox yet, and
model source written by a language model is exactly the thing you do not run in
the process holding your results: it can raise, it can loop, it can wedge jax,
and only a separate interpreter can be killed for it.
"""

import asyncio
import base64
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from discovery.evaluator import Artifact, Diagnostics
from discovery.model import Model

from examples.birthdates import data

#: where `python -m examples.birthdates` has to be run from
ROOT = Path(__file__).resolve().parents[2]

#: how much of a failed run's stderr to carry back in the error
TAIL = 2000


class FitError(RuntimeError):
    """The child could not fit the model, and this is what it said."""


@dataclass(frozen=True)
class Budget:
    """What every candidate gets, identically.

    A constant budget is what makes the numbers comparable: a model that only
    looks better because it was given more warmup has not been found to be
    better. `identified` is the KL, in nats, below which a parameter counts as
    not updated by the data.
    """

    data: str = str(data.DEFAULT)
    holdout_from: int = 1985
    warmup: int = 500
    samples: int = 500
    chains: int = 4
    seed: int = 0
    prior_draws: int = 2000
    identified: float = 0.05
    max_parameters: int = 2000


async def run(source: str, budget: Budget, timeout: float | None = None) -> dict:
    """Fit one model's source in a child process and read back its report."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "examples.birthdates.runner",
        cwd=ROOT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    request = json.dumps(asdict(budget) | {"source": source}).encode()
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
    """Everything one NUTS fit has to say about one candidate model.

    The numbers, in the order they were asked for:

     * `kl.<parameter>` -- identifiability, as the information the data added to
       each parameter's prior, in nats. `unidentified` counts those the data did
       not move: a model with a parameter it cannot learn is a worse model even
       when it predicts well.
     * `elpd_holdout` -- the model's likelihood on days it never saw, and
       `elpd_train` on the days it did.
     * `coverage95` -- how often the held-out day fell inside the 95% predictive
       interval, and `coverage_error` how far that is from 0.95.
     * `rhat.<parameter>`, `ess_tail.<parameter>` -- per parameter, with
       `rhat_max` and `ess_tail_min` for the worst of them.
     * `mcmc_seconds` -- the cost of the fit, jit compilation included.

    And the pictures: a posterior predictive check over the held-out years, and
    a calibration plot of the randomised PIT.
    """

    budget: Budget = field(default_factory=Budget)
    timeout: float | None = 600.0

    async def score(self, model: Model) -> dict:
        report = await run(model.source, self.budget, self.timeout)
        return report["metrics"] | {
            name: Artifact(name, base64.b64decode(payload))
            for name, payload in report["artifacts"].items()
        }
