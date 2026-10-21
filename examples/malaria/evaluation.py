"""One table over every proposal that fitted, local and frontier, against the incumbent.

The local sweep and the frontier searches were fitted under the same
initialisation and scored on the same held-out rows, and they ask the same
question of a proposal: what did it buy over the model it was meant to
improve. So they share a table, and every elpd in it is a paired difference
from the incumbent with its standard error (`estimators.elpd_diff`), on the
held-out fold and on the training fold. Totals alone cannot say which gaps are
real; on 35 held-out units most of them are not.

A difference more than `Z` standard errors from zero is marked: `!` in the
terminal, and `\\better` or `\\worse` in the LaTeX, which the document defines,
e.g.

    \\newcommand{\\better}[1]{\\textcolor{green!50!black}{#1}}
    \\newcommand{\\worse}[1]{\\textcolor{red!75!black}{#1}}

Ranked by training elpd, with the held-out difference beside it.

Every round that starts from the incumbent refits it, so it would otherwise
appear once per round. It appears once, after checking that every refit scored
it identically. The scratch seeds are different models and keep their rows.

    uv run python -m examples.malaria.evaluation remote/runs_pathfinder/runs_pathfinder remote/runs_agentic
    uv run python -m examples.malaria.evaluation remote/runs_pathfinder/runs_pathfinder remote/runs_agentic --latex
"""

import argparse
from pathlib import Path

from examples.malaria import frontier, sweep
from examples.malaria.__main__ import RANKED_SPLIT, compare, pointwise

#: standard errors a paired difference must clear to be marked. Two is the
#: conventional ~95%; `estimators.elpd_diff` would ask for nearer four on a fold
#: of a few dozen units, which leaves only the clearly worse rows marked.
Z = 2.0

#: the folds a difference is taken on, in column order.
FOLDS = ("holdout", "train")

#: what the table is ranked on.
TRAIN = f"posterior.{RANKED_SPLIT}.elpd_train"

#: the columns after the differences, as `(heading, latex heading, metric,
#: scale, format)`.
METRICS = (
    ("min", "min", "posterior.mcmc_seconds", 1 / 60, "{:.1f}"),
    ("pars", "pars", "posterior.parameters", 1, "{:.0f}"),
    ("ess_tail", r"ESS$_{\mathrm{tail}}$", "posterior.ess_tail_min", 1, "{:.0f}"),
    ("unid", "unid.", "posterior.unidentified", 1, "{:.0f}"),
)


def vectors(row: dict) -> tuple[tuple[float, ...], ...]:
    """A row's per-unit elpd on each of `FOLDS`."""
    found = tuple(pointwise(row["scores"], RANKED_SPLIT, fold) for fold in FOLDS)
    if any(vector is None for vector in found):
        raise SystemExit(f"{row['config']} {row['index']} has no pointwise elpd; its report predates them")
    return found  # type: ignore[return-value]


def incumbent(rows: list[dict]) -> dict:
    """The row every difference is taken against."""
    found = next((row for row in rows if row["config"] == frontier.SEED), None)
    if found is None:
        raise SystemExit("no frontier round scored the incumbent")
    return found


def combine(local: list[dict], front: list[dict]) -> list[dict]:
    """Both searches' rows, each named as the paper names it, the incumbent once.

    Frontier candidates are named by the arm that proposed them (`haiku`,
    `sol`, ...), local ones by their sweep cell (`inc/cur`, `scr/imm`, ...).
    """
    combined = [dict(row, config=row["arm"]) for row in front]
    reference = incumbent(combined)
    for row in local:
        row = dict(row, config=f"{row['start'][:3]}/{row['aperture'][:3]}")
        if row["seed"] and row["start"] == sweep.STARTS["griffin"]:
            if vectors(row) != vectors(reference):
                raise SystemExit(f"{row['run']} scored the incumbent differently from the frontier rounds")
            continue
        combined.append(row)
    return combined


def ranked(rows: list[dict]) -> list[dict]:
    """The rows in training elpd order, best first."""
    return sorted(rows, key=lambda row: -row["scores"][TRAIN])


def differences(row: dict, reference: dict) -> tuple[tuple[float, float], ...]:
    """`(difference, se)` from `reference` on each of `FOLDS`."""
    vectors(row), vectors(reference)  # refuses a report that predates the vectors, by name
    return tuple(compare(row["scores"], reference["scores"], RANKED_SPLIT, fold) for fold in FOLDS)


def verdict(difference: float, se: float) -> int:
    """1 if `difference` is more than `Z` standard errors above zero, -1 below, else 0."""
    if difference > Z * se:
        return 1
    if difference < -Z * se:
        return -1
    return 0


def signed(value: float) -> str:
    """`value` to one place with its sign, except a zero, which has none."""
    text = f"{value:+.1f}"
    return "0.0" if text in ("+0.0", "-0.0") else text


def metrics(row: dict) -> list[str]:
    """A row's `METRICS`, formatted, with `-` for anything it did not report."""
    return [
        fmt.format(row["scores"][key] * scale) if key in row["scores"] else "-"
        for _, _, key, scale, fmt in METRICS
    ]


def table(rows: list[dict]) -> str:
    """The aligned text table. `*` marks a seed, `!` a difference past `Z`."""
    reference = incumbent(rows)
    widths = (14, 17)
    head = f"{'config':>9} {'cand':>5} {'d elpd':>{widths[0]}} {'d elpd/train':>{widths[1]}} "
    head += " ".join(f"{name:>8}" for name, *_ in METRICS)
    lines = [head]
    for row in ranked(rows):
        candidate = f"{row['index']}" + ("*" if row["seed"] else " ")
        if row is reference:
            deltas = ["-"] * len(FOLDS)
        else:
            deltas = [
                f"{signed(d)}+-{se:.1f}" + ("!" if verdict(d, se) else " ")
                for d, se in differences(row, reference)
            ]
        cells = " ".join(f"{delta:>{width}}" for delta, width in zip(deltas, widths))
        lines.append(f"{row['config']:>9} {candidate:>5} {cells} " + " ".join(f"{m:>8}" for m in metrics(row)))
    return "\n".join(lines)


def marked(difference: float, se: float) -> str:
    """One difference as its two LaTeX cells, coloured if it is past `Z`."""
    cells = (signed(difference), f"{se:.1f}")
    wrap = {1: r"\better", -1: r"\worse"}.get(verdict(difference, se))
    return " & ".join(f"{wrap}{{{cell}}}" if wrap else cell for cell in cells)


def latex(rows: list[dict]) -> str:
    """The same table as a booktabs `tabular`, for pasting into the write-up.

    Each difference takes two columns joined by `\\pm`, so the errors line up.
    No `\\begin{table}` wrapper: the caption, placement and label belong to the
    document, and so do `\\better` and `\\worse`.
    """
    reference = incumbent(rows)
    lines = [
        r"\begin{tabular}{l r r@{\,$\pm$\,}l r@{\,$\pm$\,}l " + " ".join("r" for _ in METRICS) + "}",
        r"\toprule",
        r"config & cand. & \multicolumn{2}{c}{$\Delta\mathrm{elpd}$} & "
        r"\multicolumn{2}{c}{$\Delta\mathrm{elpd}_{\mathrm{train}}$} & "
        + " & ".join(head for _, head, *_ in METRICS)
        + r" \\",
        r"\midrule",
    ]
    for row in ranked(rows):
        candidate = f"{row['index']}" + (r"$^{\star}$" if row["seed"] else "")
        if row is reference:
            deltas = [r"\multicolumn{2}{c}{---}"] * len(FOLDS)
        else:
            deltas = [marked(d, se) for d, se in differences(row, reference)]
        lines.append(" & ".join([row["config"], candidate, *deltas, *metrics(row)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="examples.malaria.evaluation", description=__doc__)
    parser.add_argument("sweep", type=Path, help="the local sweep, e.g. remote/runs_pathfinder/runs_pathfinder")
    parser.add_argument("frontier", type=Path, help="the frontier runs, e.g. remote/runs_agentic")
    parser.add_argument(
        "--latex",
        action="store_true",
        help="a booktabs tabular instead of the aligned text table",
    )
    settings = parser.parse_args(argv)

    local, runs = sweep.rows(settings.sweep)
    front, arms = frontier.rows(settings.frontier)
    if not local:
        raise SystemExit(f"no scored candidates under {settings.sweep}")
    rows = combine(local, front)
    if settings.latex:
        # the coverage blocks stay out of it: how much of a search finished is
        # a sentence in the text, not a second table nobody asked for.
        print(latex(rows))
        return 0
    print(table(rows))
    print()
    print(sweep.coverage(runs))
    print()
    print(frontier.coverage(arms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
