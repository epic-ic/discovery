"""Report tests: what a run leaves on disk."""

import json

from discovery.evaluator import Artifact, Evaluation, Result
from discovery.model import Model
from discovery.report import read, record, write

SOURCE = "# @discovery:begin prior\nmu = 0.0\n# @discovery:end\n"


def evaluation(index: int = 0, error: Exception | None = None) -> Evaluation:
    return Evaluation(
        index,
        Model(SOURCE.replace("0.0", f"{index}.0")),
        (
            Result("likelihood", {"loglik": -12.5, "folds": (1.0, 2.0)}),
            Result("plots", {"trace": Artifact("trace", b"\x89PNG")}),
            Result("broken", error=error) if error else Result("rhat", {"rhat": 1.01}),
        ),
    )


def test_a_record_keeps_metrics_under_their_evaluator():
    body = record(evaluation())
    assert body["metrics"]["likelihood"] == {"loglik": -12.5, "folds": [1.0, 2.0]}
    assert body["model"].endswith("# @discovery:end\n")
    assert body["ok"] and body["errors"] == {}


def test_failures_are_recorded_not_dropped():
    body = record(evaluation(error=RuntimeError("sandbox died")))
    assert body["ok"] is False
    assert body["errors"] == {"broken": "RuntimeError: sandbox died"}


def test_write_puts_images_beside_the_report_and_points_at_them(tmp_path):
    path = write([evaluation(0), evaluation(1)], tmp_path / "run" / "report.jsonl")
    records = read(path)
    assert len(records) == 2

    reference = records[1]["metrics"]["plots"]["trace"]
    assert reference == {"artifact": "artifacts/0001-plots.trace.png", "media_type": "image/png"}
    assert (path.parent / reference["artifact"]).read_bytes() == b"\x89PNG"


def test_every_line_is_a_whole_record(tmp_path):
    path = write([evaluation(i) for i in range(3)], tmp_path / "report.jsonl")
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["index"] for line in lines] == [0, 1, 2]


def test_append_continues_a_report(tmp_path):
    path = tmp_path / "report.jsonl"
    write([evaluation(0)], path)
    write([evaluation(1)], path, append=True)
    assert [r["index"] for r in read(path)] == [0, 1]
    # without it, the report starts over
    write([evaluation(2)], path)
    assert [r["index"] for r in read(path)] == [2]


def test_a_truncated_tail_does_not_lose_the_rest(tmp_path):
    path = write([evaluation(0), evaluation(1)], tmp_path / "report.jsonl")
    with path.open("a") as handle:
        handle.write('{"index": 2, "mod')
    assert [r["index"] for r in read(path)] == [0, 1]


def test_digests_distinguish_models(tmp_path):
    path = write([evaluation(0), evaluation(1)], tmp_path / "report.jsonl")
    first, second = read(path)
    assert first["digest"] != second["digest"]
    assert len(first["digest"]) == 12
