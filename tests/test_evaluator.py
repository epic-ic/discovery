"""Evaluator and diagnostics tests.

Nothing here runs a model: an evaluator is whatever the experiment says it is,
so what the harness owes is the plumbing around it -- that the concurrency bound
is honoured, that one evaluator's failure costs one result, that the shape of
what comes back is checked at the boundary, and that diagnostics may return
pictures while plain evaluators may not.
"""

import asyncio

import pytest

from discovery.evaluator import (
    Artifact,
    Diagnostics,
    Evaluation,
    Evaluator,
    Result,
    evaluate,
)
from discovery.model import Model

SOURCE = """\
def model(x, y=None):
    # @discovery:begin prior
    slope = sample("slope", Normal(0.0, 1.0))
    # @discovery:end
    return sample("y", Normal(slope * x, 1.0), obs=y)
"""


def candidates(count: int) -> tuple[Model, ...]:
    return tuple(Model(SOURCE.replace("0.0, 1.0", f"{i}.0, 1.0")) for i in range(count))


class Likelihood(Evaluator):
    """Scores by how long the source is; a stand-in for something real."""

    async def score(self, model):
        return {"loglik": -float(len(model.source)), "folds": [1.0, 2.0, 3.0]}


class Plots(Diagnostics):
    async def score(self, model):
        return {"rhat": 1.01, "trace": Artifact("trace", b"\x89PNG...")}


class Broken(Evaluator):
    async def score(self, model):
        raise RuntimeError("sandbox died")


class Slow(Evaluator):
    """Records how many of its calls overlap, and can be made to hang."""

    def __init__(self, delay: float = 0.01, forever: bool = False) -> None:
        self.delay, self.forever = delay, forever
        self.running = self.peak = 0

    async def score(self, model):
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(3600 if self.forever else self.delay)
        finally:
            self.running -= 1
        return {"ms": self.delay * 1000}


async def test_scores_every_model_with_every_evaluator():
    evaluations = await evaluate(candidates(3), [Likelihood(), Plots()])
    assert [e.index for e in evaluations] == [0, 1, 2]
    assert [r.evaluator for r in evaluations[0].results] == ["likelihood", "plots"]
    assert all(e.ok for e in evaluations)
    assert evaluations[0].scores == {
        "likelihood.loglik": -float(len(evaluations[0].model.source)),
        "likelihood.folds": (1.0, 2.0, 3.0),
        "plots.rhat": 1.01,
    }


async def test_models_keep_their_order():
    sources = [e.model.source for e in await evaluate(candidates(5), [Likelihood()])]
    assert sources == [m.source for m in candidates(5)]


async def test_concurrency_is_bounded():
    slow = Slow()
    await evaluate(candidates(12), [slow], concurrency=4)
    assert slow.peak == 4


async def test_concurrency_is_the_only_bound():
    slow = Slow()
    await evaluate(candidates(6), [slow], concurrency=16)
    assert slow.peak == 6


async def test_concurrency_must_be_positive():
    with pytest.raises(ValueError):
        await evaluate(candidates(1), [Likelihood()], concurrency=0)


async def test_one_failure_costs_one_result():
    (evaluation,) = await evaluate(candidates(1), [Broken(), Likelihood()])
    assert not evaluation.ok
    assert isinstance(evaluation.errors["broken"], RuntimeError)
    # the evaluator that worked is unaffected, and still reports
    assert evaluation.scores["likelihood.loglik"] < 0
    assert "broken.loglik" not in evaluation.scores


async def test_a_timeout_is_a_failed_result_not_a_failed_batch():
    evaluations = await evaluate(
        candidates(2), [Slow(forever=True), Likelihood()], concurrency=4, timeout=0.05
    )
    assert all(isinstance(e.errors["slow"], TimeoutError) for e in evaluations)
    assert all("likelihood.loglik" in e.scores for e in evaluations)


async def test_diagnostics_return_artifacts_and_evaluators_do_not():
    (evaluation,) = await evaluate(candidates(1), [Plots()])
    assert evaluation.artifacts["plots.trace"].data == b"\x89PNG..."
    assert "plots.trace" not in evaluation.scores

    class Sneaky(Evaluator):
        async def score(self, model):
            return {"trace": Artifact("trace", b"")}

    (evaluation,) = await evaluate(candidates(1), [Sneaky()])
    assert isinstance(evaluation.errors["sneaky"], TypeError)


@pytest.mark.parametrize("returned", [{"x": "big"}, {"x": ["a", "b"]}, {"x": None}, ("x", 1.0)])
async def test_what_is_not_a_number_is_rejected_at_the_boundary(returned):
    class Odd(Evaluator):
        async def score(self, model):
            return returned

    (evaluation,) = await evaluate(candidates(1), [Odd()])
    assert isinstance(evaluation.errors["odd"], TypeError)


async def test_scalars_and_vectors_are_normalised():
    class Mixed(Evaluator):
        async def score(self, model):
            return {"n": 3, "flag": True, "counts": range(3)}

    (evaluation,) = await evaluate(candidates(1), [Mixed()])
    assert evaluation.scores == {"mixed.n": 3.0, "mixed.flag": 1.0, "mixed.counts": (0.0, 1.0, 2.0)}


async def test_cancellation_unwinds_the_batch():
    task = asyncio.create_task(evaluate(candidates(4), [Slow(forever=True)]))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_an_evaluation_of_nothing_is_still_well_formed():
    evaluation = Evaluation(0, Model(SOURCE), (Result("empty"),))
    assert evaluation.ok and evaluation.scores == {} and evaluation.artifacts == {}


def test_artifact_suffixes():
    assert Artifact("a", b"", "image/png").suffix == ".png"
    assert Artifact("a", b"", "image/jpeg").suffix == ".jpg"
    assert Artifact("a", b"", "image/svg+xml").suffix == ".svg"
    assert Artifact("a", b"", "application/octet-stream").suffix == ".bin"
