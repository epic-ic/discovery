"""Evaluators score models; diagnostics also draw them.

An evaluator answers one question about one model and says so in numbers: a
dictionary of scalars or vectors, named. That is the whole contract, and it is
deliberately thin, because what makes a good score is the experiment's business
and not the harness's.

A `Diagnostics` is an evaluator that may also return an `Artifact` -- a
posterior predictive plot, a trace, a residual histogram -- alongside its
numbers. It is the same call and the same pass over the model: a diagnostic
that had to be recomputed to be drawn would be a different diagnostic.

Evaluation is asynchronous for the same reason proposal is: the work happens
somewhere else, in a sandbox or on a worker, and the harness spends its time
waiting. `evaluate` runs a batch of models against a set of evaluators with a
bound on how many of those calls are in flight at once, since the sandbox, not
the event loop, is the scarce thing.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from discovery.model import Model

#: one number
Scalar = float
#: many numbers under one name: a trace, a residual series, a per-fold score
Vector = tuple[float, ...]


@dataclass(frozen=True)
class Artifact:
    """An image a diagnostic produced, carried as bytes.

    The harness does not draw and does not decode; it names the thing, keeps it
    next to the score it belongs to, and leaves it to the report to write out.
    """

    name: str
    data: bytes
    media_type: str = "image/png"

    #: subtypes whose obvious extension is not the subtype itself
    SUFFIXES = {"jpeg": ".jpg", "svg+xml": ".svg", "npz": ".npz"}

    @property
    def suffix(self) -> str:
        """A file extension for the media type, `.bin` if it has none to give."""
        subtype = self.media_type.partition("/")[2]
        return self.SUFFIXES.get(subtype) or (f".{subtype}" if subtype.isalnum() else ".bin")


#: what an evaluator may name: numbers, or -- for diagnostics only -- pictures
Value = Scalar | Vector | Artifact
Metrics = dict[str, Value]


def coerce(name: str, value: Any, *, artifacts: bool) -> Value:
    """Read one returned value as a scalar, a vector, or an artifact.

    Anything that survives `float()` is a scalar, which takes numpy and python
    numbers alike without the harness having to know about either; anything
    iterable is a vector. `artifacts` is off for a plain evaluator, so a
    picture returned by something that promised numbers is an error at the
    boundary rather than a surprise in the report.
    """
    if isinstance(value, Artifact):
        if not artifacts:
            raise TypeError(f"{name!r}: only a Diagnostics may return an artifact")
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        try:
            return tuple(float(item) for item in value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name!r}: vector holds something that is not a number") from error
    raise TypeError(f"{name!r}: expected a scalar, a vector, or an artifact, got {type(value).__name__}")


class Evaluator(ABC):
    """Scores a model.

    Subclasses implement `score`, which may do anything asynchronous it likes
    and returns a mapping of name to scalar or vector. `name` prefixes those in
    a report, so two evaluators may both call something `loglik` without
    colliding.
    """

    #: whether returned artifacts are allowed; see `Diagnostics`
    artifacts: bool = False

    @property
    def name(self) -> str:
        """How this evaluator's metrics are labelled. The class name by default."""
        return type(self).__name__.lower()

    @abstractmethod
    async def score(self, model: Model) -> dict[str, Any]:
        """Evaluate one model. Called concurrently with itself."""

    async def __call__(self, model: Model) -> Metrics:
        """Evaluate one model and check what came back."""
        returned = await self.score(model)
        if not isinstance(returned, dict):
            raise TypeError(f"{self.name}: expected a dict of metrics, got {type(returned).__name__}")
        return {name: coerce(name, value, artifacts=self.artifacts) for name, value in returned.items()}


class Diagnostics(Evaluator):
    """An evaluator that may also return images.

    Nothing else differs: a diagnostic is scored, gathered and reported exactly
    like any other evaluator, and the artifacts simply travel with the numbers
    they explain.
    """

    artifacts = True


@dataclass(frozen=True)
class Result:
    """What one evaluator said about one model, or why it said nothing."""

    evaluator: str
    metrics: Metrics = field(default_factory=dict)
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def scores(self) -> dict[str, Scalar | Vector]:
        """The numbers, without the pictures."""
        return {n: v for n, v in self.metrics.items() if not isinstance(v, Artifact)}

    @property
    def artifacts(self) -> dict[str, Artifact]:
        """The pictures, without the numbers."""
        return {n: v for n, v in self.metrics.items() if isinstance(v, Artifact)}


@dataclass(frozen=True)
class Evaluation:
    """Everything the evaluators had to say about one model.

    A model whose evaluators all failed is still an evaluation: the errors are
    the finding, and a report that dropped them would hide exactly the models
    worth looking at.
    """

    index: int
    model: Model
    results: tuple[Result, ...] = ()

    @property
    def ok(self) -> bool:
        """True when every evaluator returned."""
        return all(result.ok for result in self.results)

    def qualified(self, kind: str = "metrics") -> dict[str, Value]:
        """The named values across evaluators, each prefixed by its evaluator."""
        return {
            f"{result.evaluator}.{name}": value
            for result in self.results
            for name, value in getattr(result, kind).items()
        }

    @property
    def metrics(self) -> dict[str, Value]:
        return self.qualified("metrics")

    @property
    def scores(self) -> dict[str, Scalar | Vector]:
        return self.qualified("scores")  # type: ignore[return-value]

    @property
    def artifacts(self) -> dict[str, Artifact]:
        return self.qualified("artifacts")  # type: ignore[return-value]

    @property
    def errors(self) -> dict[str, Exception]:
        """The evaluators that failed, and what they raised."""
        return {r.evaluator: r.error for r in self.results if r.error is not None}


async def evaluate(
    models: Sequence[Model],
    evaluators: Sequence[Evaluator],
    *,
    concurrency: int = 8,
    timeout: float | None = None,
) -> tuple[Evaluation, ...]:
    """Score every model with every evaluator, at most `concurrency` at a time.

    The unit of work is one evaluator on one model, so a slow evaluator holds up
    only its own call and the models fill the pipe between them. `timeout`
    bounds each such call.

    One failure costs one result: an evaluator that raises -- or times out, or
    hands back something that is not a number -- becomes a `Result` carrying the
    exception, and the rest of the batch is unaffected. Results keep the order
    of `evaluators`, and evaluations the order of `models`, so a report reads
    the same way twice.
    """
    if concurrency < 1:
        raise ValueError(f"need at least one concurrent evaluation, got {concurrency}")
    limit = asyncio.Semaphore(concurrency)

    async def one(model: Model, evaluator: Evaluator) -> Result:
        async with limit:
            try:
                async with asyncio.timeout(timeout):
                    return Result(evaluator.name, await evaluator(model))
            except asyncio.CancelledError:
                # A cancellation from outside is not this evaluator's failure to
                # report: let it unwind the whole batch as asked.
                raise
            except Exception as error:
                return Result(evaluator.name, error=error)

    results = await asyncio.gather(
        *(one(model, evaluator) for model in models for evaluator in evaluators)
    )
    width = len(evaluators)
    return tuple(
        Evaluation(index, model, tuple(results[index * width : (index + 1) * width]))
        for index, model in enumerate(models)
    )
