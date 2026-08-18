"""One table over many runs of the same round, ranked by held-out likelihood.

`__main__.table` ranks the candidates of a single run, which is what a run needs
while it is happening. A sweep is several such runs varying two things -- where
the search started and how much of the model it was allowed to rewrite -- and
the question it asks is across them: does starting from the incumbent beat
starting from scratch, and does the wider aperture pay for itself.

Reads the reports rather than the fits, so nothing here refits anything, and a
run that died halfway still contributes the candidates it scored.

Every run in a sweep is scored on the same held-out rows, so these totals *can*
be compared -- but the totals alone cannot say by how much. On 35 units the
paired difference and its standard error are the comparison that means
something (`estimators.elpd_diff`, and `__main__.table`'s `vs best` column);
a gap of tens of nats between the top rows here is routinely inside it.

    uv run python -m examples.malaria.sweep runs_combined
    uv run python -m examples.malaria.sweep runs_combined --all
    uv run python -m examples.malaria.sweep runs_combined --latex
"""

import argparse
import json
from pathlib import Path

from examples.malaria.__main__ import RANK, RANKED_SPLIT
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
            scores = flatten(record)
            scored.append(
                {
                    "run": f"{directory.parent.name}/{directory.name}",
                    "start": start,
                    "aperture": aperture,
                    "index": record["index"],
                    "seed": record["index"] == 0,
                    "scores": scores,
                    "elpd": scores.get(RANK, float("-inf")),
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


#: the columns, as `(heading, latex heading, metric, format)`. The metric keys
#: are the flattened report keys, so `RANK` is the same string the run's own
#: table ranks on and the two cannot drift apart.
COLUMNS = (
    ("elpd", r"$\mathrm{elpd}$", RANK, "{:>10.1f}"),
    ("elpd/train", r"$\mathrm{elpd}_{\mathrm{train}}$", f"posterior.{RANKED_SPLIT}.elpd_train", "{:>10.1f}"),
    ("cover90", r"cover$_{90}$", f"posterior.{RANKED_SPLIT}.coverage90_holdout_prev_m", "{:>7.3f}"),
    ("unid", "unid.", "posterior.unidentified", "{:>4.0f}"),
    ("rhat", r"$\hat{R}$", "posterior.rhat_max", "{:>5.2f}"),
    ("ess_tail", r"ESS$_{\mathrm{tail}}$", "posterior.ess_tail_min", "{:>8.0f}"),
    ("div", "div.", "posterior.divergences", "{:>5.0f}"),
    ("mins", "min", "posterior.mcmc_seconds", "{:>6.1f}"),
    ("pars", "pars", "posterior.parameters", "{:>4.0f}"),
)


def ranked(scored: list[dict], best_only: bool) -> list[dict]:
    """The rows in `RANK` order; `best_only` keeps one per run.

    One row per run is the comparison between arms; the full listing shows the
    spread within an arm, which is what says whether its result was the start it
    was given or one lucky candidate.
    """
    order = sorted(scored, key=lambda row: -row["elpd"])
    if not best_only:
        return order
    seen: set[str] = set()
    return [row for row in order if not (row["run"] in seen or seen.add(row["run"]))]


def cells(row: dict) -> list[str]:
    """One row's metrics, formatted, with `-` for anything it did not report."""
    out = []
    for name, _, key, fmt in COLUMNS:
        if key not in row["scores"]:
            out.append("-".rjust(len(fmt.format(0))))
        else:
            value = row["scores"][key]
            out.append(fmt.format(value / 60.0 if name == "mins" else value))  # type: ignore[operator]
    return out


def table(scored: list[dict], best_only: bool) -> str:
    """The sweep, ranked by `RANK` across every run in it.

    `*` marks the model a run started from: it is not a search result, and a run
    whose best row is starred found nothing.
    """
    head = f"{'start':>10} {'aperture':>9} {'cand':>6} "
    head += " ".join(f"{name:>{len(fmt.format(0))}}" for name, _, _, fmt in COLUMNS)
    lines = [head]
    for row in ranked(scored, best_only):
        candidate = f"{row['index']:>4}" + ("*" if row["seed"] else " ")
        prefix = f"{row['start']:>10} {row['aperture']:>9} {candidate:>6} "
        lines.append(prefix + " ".join(cells(row)))
    return "\n".join(lines)


def latex(scored: list[dict], best_only: bool) -> str:
    """The same table as a booktabs `tabular`, for pasting into the write-up.

    Numbers are the ones `table` prints, formatted once in `COLUMNS` so the
    paper and the terminal cannot disagree about a value. No `\\begin{table}`
    wrapper: the caption, placement and label belong to the document, not here.
    """
    lines = [
        r"\begin{tabular}{ll r " + " ".join("r" for _ in COLUMNS) + "}",
        r"\toprule",
        "start & aperture & cand. & " + " & ".join(head for _, head, _, _ in COLUMNS) + r" \\",
        r"\midrule",
    ]
    for row in ranked(scored, best_only):
        candidate = f"{row['index']}" + (r"$^{\star}$" if row["seed"] else "")
        values = " & ".join(cell.strip() for cell in cells(row))
        lines.append(f"{row['start']} & {row['aperture']} & {candidate} & {values} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="examples.malaria.sweep", description=__doc__)
    parser.add_argument("root", type=Path, help="directory of sweeps, e.g. runs_fixed")
    parser.add_argument(
        "--all",
        action="store_true",
        help="every scored candidate, not just each run's best",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="a booktabs tabular instead of the aligned text table",
    )
    settings = parser.parse_args(argv)

    scored, runs = rows(settings.root)
    if not scored:
        raise SystemExit(f"no scored candidates under {settings.root}")
    if settings.latex:
        # the coverage block stays out of it: how much of a round finished is a
        # sentence in the text, not a second table nobody asked for.
        print(latex(scored, best_only=not settings.all))
        return 0
    print(table(scored, best_only=not settings.all))
    print()
    print(coverage(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
