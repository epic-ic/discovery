"""What a round of agents is told about the rounds before it.

    uv run python -m examples.malaria.feedback runs/beat --out /tmp/pack   # to read one yourself

A round is a directory. Once drawn it holds `proposals/`; once fitted --
by `__main__`, on this machine or on the cluster -- it also holds `report.jsonl`,
`fit.json` and `artifacts/`. `agentic --history` writes this pack from every
finished round it is pointed at and copies it into each sandbox, so a round
drawn while others are still being fitted learns from whichever have come back.

## Nothing held out

The pack is the one channel from a report to an agent, and a report holds the
held-out scores. An agent that could read them round after round would be
choosing its model on the 35 units it is ranked on, and `elpd_holdout` would
stop measuring anything. So the pack is built from an allowlist -- `TRAIN` for
numbers, the `fitted_train` table for rows -- rather than by deleting whatever
looks held out. Only rounds fitted under the same split are read: another
fold's training rows are this fold's held-out ones.

## What is in it

* `FEEDBACK.md` -- the leaderboard, the incumbent's misfit, and why the failures
  failed, small enough to read whole.
* `incumbent/` and `candidates/<rank>-*/` -- for the incumbent and the best
  `top` candidates: the rewritten blocks, a posterior summary, every training
  row beside its posterior predictive, and a residual summary. The incumbent
  also gets its fitted curves over age, drawn by msinf, for a reader that can
  see pictures.

Candidates are ranked by PSIS-LOO on the training units, paired against the
incumbent fitted in the *same* round -- same budget, same split, same machine --
so a lead is a lead and not a difference in how two rounds were run. A round
too old to carry LOO falls back to in-sample `elpd_train`, which rewards every
extra parameter, and says so.
"""

import argparse
import json
import shutil
from hashlib import sha256
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

from discovery import report
from discovery.model import Model

from examples.malaria.__main__ import compare
from examples.malaria.fit import SCORING, scoring_of
from examples.malaria.initial import STARTS
from examples.malaria.sweep import flatten

#: the split every round in this example is fitted and scored under
KIND = "study"

#: the only numbers an agent is shown, as `posterior.<key>` -- an allowlist, so a
#: metric added to the runner later stays out of the pack until it is added here.
#: `kl.<parameter>` is allowed by prefix, below.
TRAIN = (
    f"{KIND}.elpd_train",
    f"{KIND}.elpd_train_se",
    f"{KIND}.elpd_loo_train",
    f"{KIND}.elpd_loo_train_se",
    f"{KIND}.elpd_loo_train_mcse",
    f"{KIND}.elpd_train_mcse",
    f"{KIND}.loo_khat_over_0p7",
    # weighted by training LOO alone, so they say nothing about the held-out rows
    f"{KIND}.chain_weights",
    f"{KIND}.effective_chains",
    f"{KIND}.coverage90_train_inc",
    f"{KIND}.coverage90_train_prev_m",
    "parameters",
    "unidentified",
    "kl_min",
    "rhat_max",
    "ess_tail_min",
    "divergences",
    "mcmc_seconds",
)

#: posterior sites left out of the summary: one value per site or per study,
#: which is the observational model's business rather than the mechanism's
VECTORS = ("log_eir", "ft_study")

#: the age bands the residual summary groups by, in years
BANDS = (0, 1, 2, 5, 10, 15, 21)

#: how much of a failure's message is kept
ERROR_CHARS = 400

#: exceptions the machine raises rather than the proposal -- a full disk, a race
#: on a shared cache, a node out of memory. They say nothing about the model, and
#: an agent shown one would steer away from whatever it guessed caused it.
ENVIRONMENT = ("OSError", "FileNotFoundError", "PermissionError", "MemoryError")


def environmental(row: dict) -> bool:
    """Whether a failed row failed for a reason outside the proposal."""
    last = [line for reason in row["errors"].values() for line in reason.strip().splitlines()[-1:]]
    return bool(last) and all(line.split(":", 1)[0].strip() in ENVIRONMENT for line in last)


def digest(source: str) -> str:
    """`report.digest`'s hash, from the source alone."""
    return sha256(source.encode()).hexdigest()[:12]


#: the incumbent as every round fits it, at either target
INCUMBENTS = {digest(source) for (_, start), source in STARTS.items() if start == "griffin"}


def finished(paths) -> list[Path]:
    """Every round under `paths` that has been fitted, in path order.

    A path is a round itself or anything above one; a round that has been drawn
    and not yet fitted has no report and is simply not there yet.
    """
    found: set[Path] = set()
    for path in map(Path, paths):
        if (path / "report.jsonl").is_file():
            found.add(path)
        found |= {report_path.parent for report_path in path.glob("**/report.jsonl")}
    return sorted(found)


def split_of(directory: Path) -> str | None:
    """The split a round was fitted under, as `data.label` spells it."""
    meta = directory / "fit.json"
    return json.loads(meta.read_text()).get("split") if meta.is_file() else None


def candidates(paths, split: str | None = None) -> list[dict]:
    """Every candidate of every finished round, scored or not.

    With `split`, a round fitted under any other -- or that does not say -- is
    skipped with a note: its training rows may be this split's held-out ones. So
    is a round scored other than the way `fit.SCORING` says rounds are scored
    now, since its leads are in different units; `rescore` brings it up to date.
    """
    rows = []
    for directory in finished(paths):
        if split is not None and split_of(directory) != split:
            print(f"  feedback: skipping {directory}, fitted under {split_of(directory) or 'an unrecorded split'}")
            continue
        if split is not None and scoring_of(json.loads((directory / "fit.json").read_text())) != SCORING:
            print(f"  feedback: skipping {directory}, scored before {SCORING}; "
                  f"`python -m examples.malaria.rescore {directory}` brings it up to date")
            continue
        for record in report.read(directory / "report.jsonl"):
            rows.append(
                {
                    "round": directory,
                    "label": f"{directory.parent.name}/{directory.name}",
                    "index": record["index"],
                    "ok": record["ok"],
                    "digest": record["digest"],
                    "incumbent": record["digest"] in INCUMBENTS,
                    "source": record["model"],
                    "scores": flatten(record),
                    "errors": record["errors"],
                }
            )
    return rows


def lead(row: dict, reference: dict) -> tuple[str, float, float] | None:
    """`(fold, difference, se)` of `row` over `reference`, LOO if both have it."""
    for fold in ("loo", "train"):
        try:
            found = compare(row["scores"], reference["scores"], KIND, fold)
        except ValueError:
            return None
        if found is not None:
            return (fold, *found)
    return None


def ranked(rows: list[dict]) -> list[tuple[dict, tuple | None]]:
    """The fitted non-incumbent candidates, each with its lead over its round's incumbent, best first.

    A candidate refitted in a later round -- promoted to a longer budget, say --
    appears once, at its best.
    """
    reference = {row["round"]: row for row in rows if row["incumbent"] and row["ok"]}
    scored = [
        (row, lead(row, reference[row["round"]]) if row["round"] in reference else None)
        for row in rows
        if row["ok"] and not row["incumbent"]
    ]
    scored.sort(key=lambda item: -(item[1][1] if item[1] else float("-inf")))
    seen: set[str] = set()
    out = []
    for row, found in scored:
        if row["digest"] not in seen:
            seen.add(row["digest"])
            out.append((row, found))
    return out


def allowed(scores: dict) -> dict:
    """The part of a candidate's scores an agent may read."""
    keep = {}
    for key, value in scores.items():
        name = key.removeprefix("posterior.")
        if name in TRAIN or name.startswith(f"{KIND}.kl."):
            keep[name] = value
    return keep


def artifact(row: dict, name: str) -> Path | None:
    """Where one of a candidate's artifacts was written, if it was."""
    entry = row["scores"].get(f"posterior.{KIND}.{name}")
    if isinstance(entry, dict) and (row["round"] / entry["artifact"]).is_file():
        return row["round"] / entry["artifact"]
    return None


def blocks(source: str) -> str:
    """A candidate's rewritten regions, markers and all -- what a `proposal.py` holds."""
    return (
        "\n\n".join(
            f"# @discovery:begin {block.name}\n{dedent(block.body).rstrip()}\n# @discovery:end"
            for block in Model(source).blocks
        )
        + "\n"
    )


def chains(samples: Path) -> dict[str, np.ndarray]:
    """A samples file's posterior sites as `(chain, draw, ...)`, warmup dropped."""
    from msinf.compare import BY_CHAIN_KEY

    loaded = np.load(samples)
    by_chain = bool(loaded[BY_CHAIN_KEY]) if BY_CHAIN_KEY in loaded.files else True
    return {
        name: np.asarray(loaded[name], dtype=float) if by_chain else np.asarray(loaded[name], dtype=float)[None]
        for name in loaded.files
        if name != BY_CHAIN_KEY and not name.startswith("warmup__")
    }


def posterior(samples: Path, scores: dict) -> pd.DataFrame:
    """Median, 90% interval and split r-hat of every scalar the fit sampled, with its KL where scored."""
    from numpyro.diagnostics import split_gelman_rubin

    out = []
    for name, draws in chains(samples).items():
        if name in VECTORS or draws.ndim != 2:
            continue
        q05, q50, q95 = np.quantile(draws, [0.05, 0.5, 0.95])
        rhat = float(split_gelman_rubin(draws)) if draws.shape[0] > 1 else float("nan")
        out.append(
            {
                "parameter": name,
                "median": q50,
                "q05": q05,
                "q95": q95,
                "rhat": rhat,
                "kl": scores.get(f"{KIND}.kl.{name}"),
            }
        )
    return pd.DataFrame(out)


def mixing(samples: Path, frame: pd.DataFrame, weights=None, threshold: float = 1.1) -> str:
    """The training sites whose EIR the chains disagree about, one line each.

    A site whose curves fit about equally well at two transmission levels
    leaves its chains in two modes -- Griffin puts Dicko Doneguebougou near 50
    or near 320 -- and NUTS does not cross between them. Each chain's weight in
    the scored predictive is shown beside its EIR. Named from `fitted_train`, so
    a held-out site, which has no training row, never is.
    """
    from numpyro.diagnostics import split_gelman_rubin

    draws = chains(samples).get("log_eir")
    if draws is None or draws.shape[0] < 2:
        return ""
    rhat = np.asarray(split_gelman_rubin(draws))
    names = frame.drop_duplicates("site_i").set_index("site_i")
    lines = []
    for site in np.argsort(-rhat):
        if rhat[site] <= threshold or site not in names.index:
            continue
        medians = np.exp(np.median(draws[:, :, site], axis=1))
        by_chain = ", ".join(
            f"{m:.0f}" if weights is None else f"{m:.0f} ({w:.2f})" for m, w in zip(medians, weights or medians)
        )
        row = names.loc[site]
        lines.append(f"- {row['site_name']} ({row['reference']}): r-hat {rhat[site]:.2f}, annual EIR by chain {by_chain}")
    return "\n".join(lines)


def residuals(frame: pd.DataFrame) -> str:
    """Where a candidate misses the training rows, as three small tables.

    Two lenses, because a training row has two predictions. `fitted` carries the
    row's own study's effect, as the fit inferred it from that study's rows, so
    `obs/fit` away from 1 within an age band is misfit the mechanism owns. The
    predictive median, `outside90` and `pit` redraw the study effect as for a new
    study -- what a held-out study faces -- so they mix that misfit with the
    between-study spread every candidate shares, and a study-level offset there
    is mostly the study effect, not the curves.
    """
    frame = frame.assign(
        band=pd.cut(frame["age0"], BANDS, right=False, labels=[f"{a}-{b}" for a, b in zip(BANDS, BANDS[1:])]),
        outside90=(frame["observed"] < frame["pred_q05"]) | (frame["observed"] > frame["pred_q95"]),
    )

    own = "fitted" in frame

    def summary(keys):
        columns = {"rows": ("observed", "size"), "observed": ("observed", "mean")}
        if own:
            columns["fitted"] = ("fitted", "mean")
        columns |= {"predicted": ("pred_q50", "mean"), "outside90": ("outside90", "mean"), "pit": ("pit", "mean")}
        table = frame.groupby(keys, observed=True).agg(**columns)
        if own:
            table.insert(3, "obs/fit", table["observed"] / table["fitted"])
        return table.round(3).to_string()

    tails = frame.assign(tail=(frame["pit"] - 0.5).abs()).nlargest(12, "tail")
    columns = [
        "reference",
        "site_name",
        "stream",
        "age0",
        "age1",
        "observed",
        "fitted",
        "pred_q05",
        "pred_q50",
        "pred_q95",
        "pit",
    ]
    return (
        "\n\n".join(
            [
                ("Mean observed rate beside the mean fitted rate with each row's own study effect "
                 "(`fitted`, as the fit inferred it from that study's rows) and their ratio `obs/fit`: "
                 "away from 1 within an age band is misfit the mechanism is responsible for. Then, "
                 "against the predictive a new study faces -- study effect redrawn -- the mean "
                 "predictive median, the share of rows outside its 90% interval (0.10 if calibrated) "
                 "and mean PIT (0.5 if unbiased; above 0.5 means the model predicts too low). Those "
                 "three include between-study spread every candidate shares, so read a study's "
                 "offset there as mostly its study effect."
                 if own
                 else "Mean observed vs mean predictive median, the share of rows outside the 90% "
                 "predictive interval (0.10 if calibrated), and mean PIT (0.5 if unbiased; "
                 "above 0.5 means the model predicts too low)."),
                "By stream and age band:\n" + summary(["stream", "band"]),
                "By study:\n" + summary(["stream", "reference"]),
                "Rows furthest into a tail of their predictive:\n"
                + tails[[c for c in columns if c in tails]].round(3).to_string(index=False),
            ]
        )
        + "\n"
    )


def trajectories(samples: Path, table: Path, studies, directory: Path) -> None:
    """msinf's fitted curves over age for the training studies, if it can draw them.

    Griffin's mechanism only -- msinf solves dmeq with the core parameters -- so
    this is the incumbent's picture and no candidate's. A picture is an extra:
    the pack says the same thing in numbers, and failing here costs nothing.
    """
    try:
        from msinf.plotting.trajectories import plot_all_study_trajectories

        drawn = plot_all_study_trajectories(samples, table, outdir=directory, studies=list(studies))
    except Exception as error:  # noqa: BLE001 -- an extra, see above
        print(f"  feedback: no trajectory plots ({type(error).__name__}: {error})")
        return
    for outcome, path in drawn.items():
        path.rename(directory / f"trajectories_{outcome}.png")


def candidate_dir(row: dict, directory: Path, table: Path | None = None, pictures: bool = False) -> None:
    """One candidate's files: its blocks, posterior, fitted training rows and their misfit."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "blocks.py").write_text(blocks(row["source"]))
    if samples := artifact(row, "samples.npz"):
        posterior(samples, allowed(row["scores"])).round(4).to_csv(directory / "posterior.csv", index=False)
    if fitted := artifact(row, "fitted_train"):
        frame = pd.read_csv(fitted)
        frame.to_csv(directory / "fitted_train.csv", index=False)
        text = residuals(frame)
        weights = row["scores"].get(f"posterior.{KIND}.chain_weights")
        if samples and (split := mixing(samples, frame, weights)):
            text += (
                "\nSites where the chains disagree about transmission -- the curves fit them about as "
                "well at either level, and NUTS does not cross between the two. In brackets, each "
                "chain's weight in the scored predictive, set by how well it predicts the training "
                "rows:\n" + split + "\n"
            )
        (directory / "residuals.txt").write_text(text)
        if pictures and samples and table is not None:
            trajectories(samples, table, frame["study_index"].unique(), directory)
    (directory / "scores.json").write_text(json.dumps(allowed(row["scores"]), indent=2, default=str) + "\n")


def leaderboard(board: list[tuple[dict, tuple | None]]) -> str:
    """The ranked candidates as one aligned table."""
    head = (
        f"{'candidate':>34} {'vs incumbent':>15} {'pars':>5} {'unid':>5} {'rhat':>5} {'chains':>6} "
        f"{'ess_tail':>8} {'div':>4} {'min':>5}"
    )
    lines = [head]
    for row, found in board:
        s = row["scores"]
        versus = f"{found[1]:+.1f}+-{found[2]:.1f} {found[0]}" if found else "-"
        lines.append(
            f"{row['label'] + '#' + str(row['index']):>34} {versus:>15} "
            f"{s.get('posterior.parameters', float('nan')):>5.0f} {s.get('posterior.unidentified', float('nan')):>5.0f} "
            f"{s.get('posterior.rhat_max', float('nan')):>5.2f} "
            f"{s.get(f'posterior.{KIND}.effective_chains', float('nan')):>6.1f} "
            f"{s.get('posterior.ess_tail_min', float('nan')):>8.0f} "
            f"{s.get('posterior.divergences', float('nan')):>4.0f} {s.get('posterior.mcmc_seconds', float('nan')) / 60:>5.1f}"
        )
    return "\n".join(lines)


def failures(rows: list[dict], limit: int) -> str:
    """Why the candidates that never scored did not, most recent rounds first."""
    failed = [row for row in rows if not row["ok"]][::-1][:limit]
    if not failed:
        return "None.\n"
    lines = []
    for row in failed:
        reason = "; ".join(row["errors"].values()) or "no reason recorded"
        lines.append(f"- {row['label']}#{row['index']}: {reason[-ERROR_CHARS:].strip()}")
    return "\n".join(lines) + "\n"


def write(
    paths, out: Path, *, split: str, top: int = 8, table: Path | None = None, pictures: bool = True
) -> Path | None:
    """Write the pack for `split` from the finished rounds under `paths`, or `None` if there are none."""
    rows = [row for row in candidates(paths, split) if row["ok"] or not environmental(row)]
    if not rows:
        return None
    if out.exists():
        shutil.rmtree(out)
    board = ranked(rows)
    incumbent = next((row for row in rows if row["incumbent"] and row["ok"]), None)
    if incumbent is not None:
        candidate_dir(incumbent, out / "incumbent", table, pictures)
    for rank, (row, _) in enumerate(board[:top], 1):
        name = f"{rank:02d}-{row['label'].replace('/', '_')}-{row['index']}"
        candidate_dir(row, out / "candidates" / name)

    fold = {"loo", "train"} & {found[0] for _, found in board if found}
    text = FEEDBACK.format(
        rounds=len({row["round"] for row in rows}),
        fitted=sum(1 for row in rows if row["ok"]),
        failed=sum(1 for row in rows if not row["ok"]),
        score="leave-one-out (PSIS, per chain)"
        if fold == {"loo"}
        else "in-sample elpd_train (no LOO in these rounds)"
        if fold == {"train"}
        else "PSIS-LOO where recorded, else in-sample elpd_train",
        directories=f"The top {min(top, len(board))} have their own directory under `candidates/`." if board else "",
        board=leaderboard(board[:top]) if board else "(no candidate has been fitted yet)",
        incumbent=(out / "incumbent" / "residuals.txt").read_text()
        if (out / "incumbent" / "residuals.txt").is_file()
        else "(not fitted yet)\n",
        failures=failures(rows, top * 2),
    )
    (out / "FEEDBACK.md").write_text(text)
    return out


#: the pack's front page. Training rows only, and it says so, because an agent
#: that reads "the incumbent scores -1830" will otherwise assume it is the score.
FEEDBACK = """\
# What earlier rounds found

{rounds} round(s) fitted so far: {fitted} models scored, the incumbent's fits
included, and {failed} failed a gate or the fit. Every number here comes from the **training rows** -- the rows you
can see in `data/`. The held-out rows that decide the ranking are not described
anywhere in this directory, by design.

## The best candidates so far

Ranked by {score} on the training units, as a paired difference over the
incumbent (Griffin) fitted in the same round, in nats, +- its standard error.
More is better; a lead under about twice its error is not a lead. The error
includes how much each score moves with which chains happened to run, not only
which units were drawn. `unid` counts parameters the data did not move from
their prior. {directories}

Leave-one-out here leaves a unit out given the rest of its study, which is what
removing one unit from this likelihood means: a study's effect is shared by its
units and the others still inform it. Chains are not pooled. This posterior has
modes that NUTS does not cross -- the incumbent's chains split between two
transmission levels at one site -- so `rhat` above 1.1 says the model has more
than one way to fit these rows, not that it was fitted badly. Each chain's
draws are scored on their own and the chains weighted by stacking on the
training rows: `chains` is how many chains' worth of weight the score rests on.

{board}

## Where the incumbent misses the training rows

Its full table is `incumbent/fitted_train.csv`: every training row's observed
rate beside the 5/50/95% posterior predictive, with study effects redrawn as for
a new study, which is what a held-out study faces. `incumbent/posterior.csv`
is its posterior, `incumbent/blocks.py` its blocks.

{incumbent}
## Candidates that never scored

{failures}
## Using this

`candidates/<rank>-*/blocks.py` is in `proposal.py`'s format, so a good idea
can be carried forward and changed rather than rewritten. Nothing here obliges
you to: a candidate is scored on held-out studies, and structure that fits the
training rows by spending parameters on them is what scores worse there.
"""


def main(argv: list[str] | None = None) -> int:
    from examples.malaria import data

    parser = argparse.ArgumentParser(prog="examples.malaria.feedback", description=__doc__)
    parser.add_argument("history", nargs="+", type=Path, help="rounds, or directories above them")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=8, help="candidates given a directory of their own")
    parser.add_argument("--holdout-seed", type=int, default=42)
    parser.add_argument("--n-studies", type=int, default=3)
    parser.add_argument("--no-pictures", action="store_true")
    settings = parser.parse_args(argv)
    written = write(
        settings.history,
        settings.out,
        split=data.label(KIND, settings.holdout_seed, settings.n_studies),
        top=settings.top,
        table=data.DEFAULT_TABLE,
        pictures=not settings.no_pictures,
    )
    print(f"feedback: {written}" if written else "feedback: no finished round under those paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
