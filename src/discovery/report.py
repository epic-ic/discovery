"""Writing down what an evaluation found.

A report is a jsonl file: one line per model, holding the model itself, the
scores its evaluators returned and whatever the diagnostics drew. Jsonl because
a record is naturally nested -- a vector is a list, an evaluator's metrics are
an object -- and because a line is complete on its own, so a run can append as
it goes and a report of a run that died halfway is still a report.

Images are the one thing that does not fit on a line. They are written beside
the report and the record points at them, so a row still carries the number,
the trace behind it and the picture of it.
"""

import json
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from discovery.evaluator import Artifact, Evaluation, Value


def digest(evaluation: Evaluation) -> str:
    """A short content hash of the model, for telling two records apart."""
    return sha256(evaluation.model.source.encode()).hexdigest()[:12]


def value(item: Value, artifacts: dict[int, str]) -> object:
    """One metric as it appears in the record: a number, a list, or a path.

    An image that has not been written anywhere is named rather than located,
    so a record can be built without a directory to put it in.
    """
    if isinstance(item, Artifact):
        return {"artifact": artifacts.get(id(item), item.name), "media_type": item.media_type}
    return list(item) if isinstance(item, tuple) else item


def record(evaluation: Evaluation, artifacts: dict[int, str] | None = None) -> dict[str, object]:
    """One evaluation as a json object.

    Metrics stay grouped by the evaluator that produced them: two evaluators may
    both report a `loglik` and neither has to know about the other. Failures are
    recorded rather than dropped -- an evaluator that raised is the finding, and
    a report that omitted it would hide exactly the models worth looking at.
    """
    artifacts = artifacts or {}
    return {
        "index": evaluation.index,
        "digest": digest(evaluation),
        "ok": evaluation.ok,
        "model": evaluation.model.source,
        "metrics": {
            result.evaluator: {name: value(item, artifacts) for name, item in result.metrics.items()}
            for result in evaluation.results
        },
        "errors": {who: f"{type(why).__name__}: {why}" for who, why in evaluation.errors.items()},
    }


def write(
    evaluations: Iterable[Evaluation],
    path: str | Path,
    *,
    append: bool = False,
) -> Path:
    """Write the report, and return its path.

    Images go to an `artifacts/` directory beside the report, named after the
    record they belong to, and the record holds their path relative to that
    report, so the whole thing can be moved or archived together. `append` adds
    to an existing report instead of replacing it, which is what a search that
    reports each round as it finishes wants.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w") as handle:
        for evaluation in evaluations:
            written: dict[int, str] = {}
            for label, artifact in evaluation.artifacts.items():
                relative = Path("artifacts") / f"{evaluation.index:04d}-{label}{artifact.suffix}"
                (path.parent / relative).parent.mkdir(parents=True, exist_ok=True)
                (path.parent / relative).write_bytes(artifact.data)
                written[id(artifact)] = str(relative)
            handle.write(json.dumps(record(evaluation, written)) + "\n")
    return path


def read(path: str | Path) -> tuple[dict[str, object], ...]:
    """Read a report back, skipping a trailing partial line.

    A run killed mid-write leaves one truncated record; everything before it is
    still good, and that is the point of the format.
    """
    records: list[dict[str, object]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return tuple(records)
