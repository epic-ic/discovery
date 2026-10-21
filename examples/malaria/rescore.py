"""Score fitted rounds again from their archived chains, under the current scoring.

    uv run python -m examples.malaria.rescore runs/beat
    uv run python -m examples.malaria.rescore runs/beat/refit-01 --concurrency 4

A round's report is only comparable with another's if both were scored the same
way, and the scoring has changed under rounds that took hours of sampling: the
chains are now stacked rather than pooled, and training LOO leaves a unit out
given the rest of its study (see `msinf.evaluate.score`). The chains themselves
are archived beside every report (`artifacts/*samples.npz`), so nothing needs
refitting -- every fitted candidate goes back through the runner, which scores
the archive instead of sampling, and the report is rewritten from what comes back.

What scoring cannot recompute is kept from the original: divergences, which the
archive does not hold, and the timings, which describe the fit. Everything else
-- every elpd, the coverage, the fitted tables and their pictures, r-hat, ESS
and the KL -- is recomputed from the same draws.

The original report is kept as `report.pooled.jsonl`, and the artifacts the new
one replaces under `artifacts.pooled/`, the first time a round is re-scored; a
round whose `fit.json` already records the current scoring is skipped.
"""

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path

from discovery import report

from examples.malaria import feedback
from examples.malaria.fit import SCORING, TAIL, Budget, FitError, artifact, run

#: run-level metrics a re-score cannot recompute, kept from the original fit
KEPT = ("divergences", "mcmc_seconds", "total_seconds")

#: per-split metrics likewise, as `<split>.<name>`
KEPT_PER_SPLIT = ("n_divergent", "mcmc_seconds", "fit_seconds")

#: one candidate's re-score, which is mostly the ODE-bound pass over every draw
TIMEOUT = 3600.0


def budget_of(meta: dict) -> Budget:
    """The budget a round was fitted with, as far as scoring it again needs."""
    return Budget(
        splits=tuple(meta["splits"]),
        holdout_seed=meta["holdout_seed"],
        n_studies=meta["n_studies"],
        warmup=meta["warmup"],
        samples=meta["samples"],
        chains=meta["chains"],
        seed=meta["seed"],
        inc_max=meta.get("inc_max", Budget.inc_max),
    )


def merge(record: dict, rescored: dict, splits: tuple[str, ...], written: dict[str, dict]) -> dict:
    """`record` with the re-scored metrics in place of its old ones.

    `rescored` is the runner's report; `written` maps an artifact's metric name
    to the record entry pointing at where it now is.
    """
    old = record["metrics"]["posterior"]
    new = {name: list(value) if isinstance(value, tuple) else value for name, value in rescored["metrics"].items()}
    kept = {name: old[name] for name in KEPT if name in old}
    kept |= {f"{kind}.{name}": old[f"{kind}.{name}"] for kind in splits for name in KEPT_PER_SPLIT
             if f"{kind}.{name}" in old}
    # the archive itself is not re-emitted, so its entry stays as it was
    archives = {name: value for name, value in old.items() if name.endswith(".samples.npz")}
    return record | {"metrics": record["metrics"] | {"posterior": old | new | written | kept | archives}}


async def rescore(directory: Path, concurrency: int, force: bool) -> Path | None:
    """Re-score every fitted candidate of one round, and rewrite its report."""
    meta_path = directory / "fit.json"
    meta = json.loads(meta_path.read_text())
    if meta.get("scoring") == SCORING and not force:
        print(f"{directory}: already scored as {SCORING}")
        return None
    budget = budget_of(meta)
    records = list(report.read(directory / "report.jsonl"))
    gate = asyncio.Semaphore(concurrency)

    async def one(record: dict) -> dict:
        metrics = record["metrics"].get("posterior", {})
        archive = {
            kind: str(directory / metrics[f"{kind}.samples.npz"]["artifact"])
            for kind in budget.splits
            if isinstance(metrics.get(f"{kind}.samples.npz"), dict)
        }
        if not record["ok"] or set(archive) != set(budget.splits):
            return record
        async with gate:
            started = time.perf_counter()
            try:
                rescored = await run(record["model"], budget, TIMEOUT, samples=archive)
            except FitError as error:
                print(f"  #{record['index']}: not re-scored -- {str(error)[-TAIL:]}")
                return record
        written = {}
        for name, payload in rescored["artifacts"].items():
            key, item = artifact(name, payload)
            relative = Path("artifacts") / f"{record['index']:04d}-posterior.{key}{item.suffix}"
            backup = directory / "artifacts.pooled" / relative.name
            if (directory / relative).is_file() and not backup.exists():
                backup.parent.mkdir(exist_ok=True)
                shutil.copy2(directory / relative, backup)
            (directory / relative).write_bytes(item.data)
            written[key] = {"artifact": str(relative), "media_type": item.media_type}
        merged = merge(record, rescored, budget.splits, written)
        new = merged["metrics"]["posterior"]
        print(
            f"  #{record['index']}: held-out {metrics.get('study.elpd_holdout', float('nan')):.1f} -> "
            f"{new.get('study.elpd_holdout', float('nan')):.1f}, training LOO "
            f"{metrics.get('study.elpd_loo_train', float('nan')):.1f} -> {new.get('study.elpd_loo_train', float('nan')):.1f}"
            f" ({time.perf_counter() - started:.0f}s)"
        )
        return merged

    print(f"{directory}: re-scoring {sum(1 for r in records if r['ok'])} fitted candidates")
    merged = await asyncio.gather(*(one(record) for record in records))

    backup = directory / "report.pooled.jsonl"
    if not backup.exists():
        shutil.copy2(directory / "report.jsonl", backup)
    with (directory / "report.jsonl").open("w") as handle:
        for record in merged:
            handle.write(json.dumps(record) + "\n")
    meta_path.write_text(json.dumps(meta | {"scoring": SCORING, "rescored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                                    indent=2) + "\n")
    return directory / "report.jsonl"


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="examples.malaria.rescore", description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="fitted rounds, or directories above them")
    parser.add_argument("--concurrency", type=int, default=4, help="candidates scored at once")
    parser.add_argument("--force", action="store_true", help="re-score a round already under the current scoring")
    settings = parser.parse_args(argv)
    rounds = feedback.finished(settings.paths)
    if not rounds:
        raise SystemExit(f"no fitted round under {', '.join(map(str, settings.paths))}")
    for directory in rounds:
        if not (directory / "fit.json").is_file():
            print(f"{directory}: no fit.json, so no record of how it was fitted; skipped")
            continue
        if written := await rescore(directory, settings.concurrency, settings.force):
            print(f"report: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
