"""The local search sweep's reports, with each candidate labelled by its arm.

A sweep is several runs of the same round varying two things -- where the
search started and how much of the model it was allowed to rewrite -- and the
question it asks is across them: does starting from the incumbent beat
starting from scratch, and does the wider aperture pay for itself. `rows`
reads every run and labels each candidate with its arm; `evaluation` puts
those rows in one table with the frontier searches'.

Reads the reports rather than the fits, so nothing here refits anything, and a
run that died halfway still contributes the candidates it scored.
"""

import json
from pathlib import Path

from discovery import report

#: what the starting models are called on disk, and what they are in the sweep.
#: `griffin` is the published 2014 model, so a run starting there is the search
#: given the incumbent to improve rather than to rediscover.
STARTS = {"griffin": "incumbent", "scratch": "scratch"}


def flatten(record: dict) -> dict[str, object]:
    """A record's metrics as `evaluator.metric`, the keys `RANK` is written in."""
    return {
        f"{evaluator}.{name}": item
        for evaluator, metrics in record["metrics"].items()
        for name, item in metrics.items()
    }


def rows(root: Path) -> tuple[list[dict], list[dict]]:
    """Every scored candidate under `root`, and one summary per run.

    A run is a directory holding `report.jsonl`; its `proposals/round.json` says
    what the round was, and its parent directory names the sweep it belongs to
    -- which is where the gate variant lives, since the gate is a property of
    how the sweep was launched and is not written into the round.
    """
    scored: list[dict] = []
    runs: list[dict] = []
    for path in sorted(root.glob("*/*/report.jsonl")):
        directory = path.parent
        meta_path = directory / "proposals" / "round.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        aperture = meta.get("target") or directory.name.split("-")[0]
        start = STARTS.get(meta.get("start", ""), meta.get("start", "?"))
        records = report.read(path)
        for record in records:
            if not record["ok"]:
                continue
            scored.append(
                {
                    "run": f"{directory.parent.name}/{directory.name}",
                    "start": start,
                    "aperture": aperture,
                    "index": record["index"],
                    "seed": record["index"] == 0,
                    "scores": flatten(record),
                }
            )
        runs.append(
            {
                "run": f"{directory.parent.name}/{directory.name}",
                "start": start,
                "aperture": aperture,
                "proposed": meta.get("candidates", len(records)),
                "reported": len(records),
                "ok": sum(1 for r in records if r["ok"]),
            }
        )
    return scored, runs


def coverage(runs: list[dict]) -> str:
    """What each run actually managed to score, which the ranking cannot show.

    A run with one usable candidate out of nine is not a better or worse arm
    than one with four; it is an arm that mostly did not finish, and reading its
    single row as the arm's answer is the mistake this block exists to prevent.
    """
    lines = [f"{'start':>10} {'aperture':>9} {'ok':>4} {'fitted':>7} {'proposed':>9}"]
    for run in sorted(runs, key=lambda r: (r["aperture"], r["start"])):
        lines.append(
            f"{run['start']:>10} {run['aperture']:>9} "
            f"{run['ok']:>4} {run['reported']:>7} {run['proposed']:>9}"
        )
    return "\n".join(lines)
