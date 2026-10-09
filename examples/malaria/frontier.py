"""The frontier searches' reports, with each candidate tied to the model that proposed it.

`sweep` reads the arms of the local search -- where the search started and how
much of the model it could rewrite -- from runs that each hold one arm. The
frontier searches vary something else: which frontier model proposed, and under
which harness. Two agents' worth of proposals were fitted together in one
combined round, so the arm is not the run here, it is the candidate's
provenance, and provenance is not written into the round.

It is recoverable, because a combined round's `proposals/` directory is the
concatenation of the rounds it was combined from, byte for byte. `provenance`
hashes the files and looks each one up in the arms it could have come from,
which ties a candidate index to the model that wrote it without anyone having
to keep a hand-written list in step. The seed model belongs to no arm: it is
the incumbent, and it appears in every source directory unchanged.

Reads the reports rather than the fits, so nothing here refits anything. A
candidate the round could not fit has no row; where a `refit` run has since
scored those candidates, its records are preferred over the original round's,
which is what an arm whose failures were a harness fault rather than a
modelling one needs. `evaluation` puts these rows in one table with the local
sweep's.
"""

import json
from hashlib import sha256
from pathlib import Path

from examples.malaria.sweep import flatten
from discovery import report

#: what a frontier search was: the round that fitted the candidates, the
#: proposals that round holds, and the arms it was combined from. `sources` is
#: ordered as the arms appear in the combined directory, but nothing depends on
#: that order -- provenance is decided by content. `refit`, where present, is a
#: later round over the candidates the first one failed to fit.
ARMS = (
    {
        "harness": "claude code",
        "report": "runs_agentic/18-claude-haiku-opus-16-pathfinder/report.jsonl",
        "proposals": "12-claude-haiku-opus-16/curves-griffin/proposals",
        "sources": {
            "haiku": "10-claude-haiku-8/curves-griffin/proposals",
            "opus": "11-claude-opus-8/curves-griffin/proposals",
        },
        "refit": None,
    },
    {
        "harness": "codex",
        "report": "17-codex-luna-sol-16-pathfinder/report.jsonl",
        "proposals": "07-codex-luna-sol-16/curves-griffin/proposals",
        "sources": {
            "luna": (
                "03-codex-danger-full/curves-griffin/proposals",
                "04-codex-danger-7/curves-griffin/proposals",
            ),
            "sol": (
                "05-codex-sol/curves-griffin/proposals",
                "06-codex-sol-7/curves-griffin/proposals",
            ),
        },
        "refit": "runs_agentic/18-codex-luna-sol-11-pathfinder-fixed/report.jsonl",
    },
)

#: the model a run started from, which every arm shares and no arm proposed.
SEED = "incumbent"


def directories(paths: str | tuple[str, ...]) -> tuple[str, ...]:
    """An arm's source rounds; one round may be written without the tuple."""
    return (paths,) if isinstance(paths, str) else paths


def version(root: Path, paths: str | tuple[str, ...]) -> str:
    """What model an arm's rounds asked for, as their `round.json` recorded it.

    `requested_model` rather than `llm_model`: a harness that will not say what
    it served writes a placeholder into the latter, and the name the paper wants
    is the one the run asked for either way. An arm's rounds differ in how they
    were launched, not in what they asked for, so the first one speaks for all.
    """
    meta = json.loads((root / directories(paths)[0] / "round.json").read_text())
    requested = meta.get("requested_model") or meta.get("llm_model") or "?"
    return requested.split("+")[0]


def hashes(directory: Path) -> dict[str, str]:
    """Every proposal in a directory, by name, as a content hash."""
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*.py"))}


def provenance(root: Path, arm: dict) -> dict[int, str]:
    """Which arm proposed each candidate of a combined round.

    A file that appears in more than one source is the seed: the arms were all
    given the same incumbent to improve, so its hash is the one thing they have
    in common. A file that appears in none of them is a round that was not
    combined from what we think it was, and saying so is better than guessing.
    """
    combined = hashes(root / arm["proposals"])
    sources = {
        label: {digest for path in ([paths] if isinstance(paths, str) else paths) for digest in hashes(root / path).values()}
        for label, paths in arm["sources"].items()
    }
    found: dict[int, str] = {}
    for name, digest in combined.items():
        owners = [label for label, digests in sources.items() if digest in digests]
        if not owners:
            raise SystemExit(f"{arm['proposals']}/{name} is in none of {list(sources)}")
        found[int(name.split("-")[0])] = SEED if len(owners) > 1 else owners[0]
    return found


def records(root: Path, arm: dict) -> dict[int, dict]:
    """An arm's report, with a refit's records preferred where it has them.

    A refit is a round over a subset of the candidates, so it numbers its own
    from zero and its indices are not the arm's. Its proposals keep the names
    they were given in the round they came from, and a round numbers candidates
    in the order of that directory, so the names in order are the arm's indices
    for its records -- which is the only thing tying the two reports together,
    since a repaired proposal is not the same source and so not the same digest.

    Its records are preferred whether or not they fitted: a candidate the refit
    could not fit either has failed for the reason the refit gives, not for the
    one the original round gave.
    """
    found = {record["index"]: record for record in report.read(root / arm["report"])}
    refit = root / arm["refit"] if arm["refit"] else None
    if refit is None or not refit.exists():
        return found
    indices = [int(path.name.split("-")[0]) for path in sorted((refit.parent / "proposals").glob("*.py"))]
    for record in report.read(refit):
        if record["index"] >= len(indices):
            raise SystemExit(f"{arm['refit']} scored {record['index']} but holds {len(indices)} proposals")
        found[indices[record["index"]]] = record
    return found


def rows(root: Path) -> tuple[list[dict], list[dict]]:
    """Every scored candidate of every arm, and one summary per arm.

    Candidates are named by the model that proposed them rather than by their
    round, so the seed -- fitted once per round, identically -- is one row here
    and not one per harness. `arm` is the short name `ARMS` gives that model,
    or `SEED`; `model` is the version the arm's rounds asked for.
    """
    scored: list[dict] = []
    arms: list[dict] = []
    seen: set[str] = set()
    for arm in ARMS:
        who = provenance(root, arm)
        found = records(root, arm)
        versions = {SEED: SEED} | {label: version(root, paths) for label, paths in arm["sources"].items()}
        for index, record in sorted(found.items()):
            model = who.get(index, "?")
            if not record["ok"]:
                continue
            if model == SEED and record["digest"] in seen:
                continue
            seen.add(record["digest"])
            scored.append(
                {
                    # the seed was fitted by whichever round reached it first,
                    # but it is nobody's proposal and no harness's result.
                    "harness": "-" if model == SEED else arm["harness"],
                    "arm": model,
                    "model": versions[model],
                    "index": index,
                    "seed": model == SEED,
                    "scores": flatten(record),
                }
            )
        for label in arm["sources"]:
            proposed = [index for index, name in who.items() if name == label]
            arms.append(
                {
                    "harness": arm["harness"],
                    "model": versions[label],
                    "proposed": len(proposed),
                    "ok": sum(1 for index in proposed if found.get(index, {}).get("ok")),
                }
            )
    return scored, arms


def coverage(arms: list[dict]) -> str:
    """What each arm actually managed to score, which the ranking cannot show.

    An arm whose proposals mostly failed to fit is not a worse modeller than one
    whose proposals fitted; it is an arm we know less about, and reading its
    surviving rows as its answer is the mistake this block exists to prevent.
    """
    lines = [f"{'harness':>12} {'model':>28} {'ok':>4} {'proposed':>9}"]
    for arm in arms:
        lines.append(f"{arm['harness']:>12} {arm['model']:>28} {arm['ok']:>4} {arm['proposed']:>9}")
    return "\n".join(lines)
