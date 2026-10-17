"""Every fitted round under a directory, each candidate against its round's incumbent.

    uv run python -m examples.malaria.rounds runs/beat
    uv run python -m examples.malaria.rounds runs/beat --promote 3 --to runs/beat/confirm-01

The table for whoever runs the search, and the one place outside the reports
that reads the held-out scores -- `feedback` shows agents the training columns
only. Every difference is paired against the incumbent fitted in the *same*
round, so a round fitted at another budget or on another machine is never
compared with somebody else's Griffin. Rounds are grouped by how they were
fitted (`fit.json`), and each group is ranked by training LOO -- the score the
agents were shown -- with the held-out difference beside it.

`--promote` is the second stage. The best few by training LOO, never by the
held-out fold, are copied with the incumbent into a new round, to be refitted at
a longer budget before anything is called a win:

    uv run python -m examples.malaria --proposals runs/beat/confirm-01/proposals \\
        --out runs/beat/confirm-01 --warmup 400 --samples 400

Selecting on the held-out fold and then confirming on it would be confirming the
selection, which is why the order here is training LOO even though the column
that decides is the held-out one.
"""

import argparse
import json
from pathlib import Path

from examples.estimators import elpd_diff
from examples.malaria import data, feedback
from examples.malaria.__main__ import pointwise
from examples.malaria.evaluation import METRICS, metrics, signed, verdict

#: the folds a difference is shown on, in column order
FOLDS = ("holdout", "train", "loo")


def fit_of(directory: Path) -> dict:
    """How a round was fitted, from `fit.json`; empty for a round that predates it."""
    path = directory / "fit.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def group(row: dict) -> str:
    """The rounds a row may be ranked against: same split, same budget."""
    meta = fit_of(row["round"])
    if not meta:
        return "unrecorded split and budget"
    return f"{meta['split']}, {meta['warmup']}+{meta['samples']} x {meta['chains']} chains"


def cell(row: dict, reference: dict | None, fold: str) -> str:
    """One paired difference as `d+-se`, with `!` past `evaluation.Z`."""
    if reference is None or row is reference:
        return "-"
    mine, theirs = pointwise(row["scores"], feedback.KIND, fold), pointwise(reference["scores"], feedback.KIND, fold)
    if mine is None or theirs is None:
        return "-"
    difference, se = elpd_diff(mine, theirs)
    return f"{signed(difference)}+-{se:.1f}" + ("!" if verdict(difference, se) else " ")


def table(rows: list[dict]) -> str:
    """One block per group: the incumbents first, then the candidates by training LOO lead."""
    reference = {row["round"]: row for row in rows if row["incumbent"] and row["ok"]}
    lead = {id(row): found for row, found in feedback.ranked(rows)}
    blocks = []
    for name in sorted({group(row) for row in rows}):
        members = [row for row in rows if row["ok"] and group(row) == name]
        members.sort(key=lambda row: (not row["incumbent"], -(lead.get(id(row)) or (0, float("-inf")))[1]))
        head = f"{'candidate':>34} " + " ".join(f"{'d elpd/' + fold:>16}" for fold in FOLDS)
        head += " " + " ".join(f"{label:>8}" for label, *_ in METRICS)
        lines = [f"## {name}", head]
        for row in members:
            label = f"{row['label']}#{row['index']}" + ("*" if row["incumbent"] else " ")
            cells = " ".join(f"{cell(row, reference.get(row['round']), fold):>16}" for fold in FOLDS)
            lines.append(f"{label:>34} {cells} " + " ".join(f"{m:>8}" for m in metrics(row)))
        failed = sum(1 for row in rows if not row["ok"] and group(row) == name)
        lines.append(f"\n{len(members)} fitted, {failed} failed. * the incumbent each round was fitted beside;")
        lines.append("differences are against it, in nats, and ! marks one more than two errors from zero.")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def promote(rows: list[dict], count: int, to: Path) -> Path:
    """Copy the best `count` candidates by training LOO, and the incumbent, into a new round."""
    incumbent = next((row for row in rows if row["incumbent"]), None)
    if incumbent is None:
        raise SystemExit("no round under those paths fitted the incumbent; nothing to pair a confirmation against")
    chosen = [row for row, found in feedback.ranked(rows) if found][:count]
    if not chosen:
        raise SystemExit("no candidate has a training score to rank on")
    directory = to / "proposals"
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.py"):
        old.unlink()
    meta = json.loads((incumbent["round"] / "proposals" / "round.json").read_text()) if (
        incumbent["round"] / "proposals" / "round.json"
    ).is_file() else {}
    start = meta.get("start", "griffin")
    (directory / f"0000-{start}.py").write_text(incumbent["source"])
    for index, row in enumerate(chosen, 1):
        (directory / f"{index:04d}-{start}.py").write_text(row["source"])
    meta |= {
        "candidates": len(chosen) + 1,
        "promoted_from": [{"round": str(row["round"]), "index": row["index"]} for row in chosen],
    }
    (directory / "round.json").write_text(json.dumps(meta, indent=2) + "\n")
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="examples.malaria.rounds", description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="rounds, or directories above them")
    parser.add_argument("--promote", type=int, default=0, help="copy the best N by training LOO into --to")
    parser.add_argument("--to", type=Path, default=None, help="the new round --promote writes")
    parser.add_argument("--holdout-seed", type=int, default=42, help="the split --promote ranks within")
    parser.add_argument("--n-studies", type=int, default=3)
    settings = parser.parse_args(argv)

    rows = feedback.candidates(settings.paths)
    if not rows:
        raise SystemExit(f"no fitted round under {', '.join(map(str, settings.paths))}")
    print(table(rows))
    if settings.promote:
        if settings.to is None:
            raise SystemExit("--promote needs --to, the round to write")
        # within the search's own split: a lead measured on another fold's
        # training rows was measured on some of this fold's held-out ones
        split = data.label(feedback.KIND, settings.holdout_seed, settings.n_studies)
        written = promote(feedback.candidates(settings.paths, split), settings.promote, settings.to)
        print(f"\npromoted: {written}")
        print(f"fit it with: uv run python -m examples.malaria --proposals {written} --out {settings.to} "
              "--warmup 400 --samples 400")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
