"""Search for the process behind twenty years of daily birth counts.

    uv run --group examples python -m examples.birthdates --url http://host:8000/v1

One round: ask N agents for a generating process, fit every proposal and the
model they started from with the same NUTS budget, score them all, write the
report. `--no-search` skips the language model and fits only the starting model,
which is the fastest way to see the whole evaluation path work.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

from discovery.analysis import Analysis
from discovery.evaluator import evaluate
from discovery.model import Model
from discovery.report import write
from discovery.search import AgentSearch, Decode, IIDSearch, models

from examples.birthdates import analysis as setup
from examples.birthdates import data
from examples.birthdates.fit import Budget, Posterior
from examples.birthdates.initial import BRIEF, SOURCE

#: the score models are ranked by
RANK = "posterior.elpd_holdout"


def options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="examples.birthdates", description=__doc__)
    llm = parser.add_argument_group("inference server")
    llm.add_argument(
        "--url",
        default=os.environ.get("DISCOVERY_URL"),
        help="OpenAI-compatible base url, e.g. http://host:8000/v1 [$DISCOVERY_URL]",
    )
    llm.add_argument("--llm-model", default=None, help="defaults to whatever the server serves")
    llm.add_argument("--api-key", default=os.environ.get("DISCOVERY_API_KEY", "EMPTY"))
    llm.add_argument("--agents", type=int, default=8, help="proposals drawn in one request")
    llm.add_argument("--temperature", type=float, default=1.0)
    llm.add_argument("--top-p", type=float, default=0.95)
    llm.add_argument("--top-k", type=int, default=50)
    llm.add_argument("--max-tokens", type=int, default=4096)
    llm.add_argument(
        "--thinking-token-budget",
        type=int,
        default=512,
        help="bound the reasoning; left alone it can spend max-tokens and return nothing",
    )
    llm.add_argument("--search-timeout", type=float, default=900.0)
    llm.add_argument("--no-search", action="store_true", help="fit only the starting model")

    tools = parser.add_argument_group("data analysis at proposal time")
    tools.add_argument(
        "--analysis",
        action="store_true",
        help="let agents explore the training data before proposing; needs a server started "
        "with --enable-auto-tool-choice and --tool-call-parser",
    )
    tools.add_argument("--analysis-workers", type=int, default=4, help="snippets running at once")
    tools.add_argument("--tool-budget", type=int, default=6, help="snippets one agent may run")
    tools.add_argument("--analysis-timeout", type=float, default=20.0, help="seconds per snippet")

    fit = parser.add_argument_group("fit, identical for every candidate")
    fit.add_argument("--data", default=str(data.DEFAULT), type=str)
    fit.add_argument("--holdout-from", type=int, default=1985, help="first held-out year")
    fit.add_argument("--warmup", type=int, default=500)
    fit.add_argument("--samples", type=int, default=500, help="post-warmup draws per chain")
    fit.add_argument("--chains", type=int, default=4)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--max-parameters", type=int, default=2000)
    fit.add_argument("--concurrency", type=int, default=2, help="fits in flight at once")
    fit.add_argument("--fit-timeout", type=float, default=600.0)

    parser.add_argument("--out", type=Path, default=Path("runs/birthdates"))
    return parser.parse_args(argv)


def served(url: str) -> str:
    """The first model the server offers, so the url is the only thing to know."""
    with urllib.request.urlopen(f"{url.rstrip('/')}/models", timeout=30) as response:
        offered = json.load(response)["data"]
    if not offered:
        raise SystemExit(f"{url} serves no models")
    return offered[0]["id"]


async def propose(settings: argparse.Namespace, start: Model, brief: str) -> tuple[Model, ...]:
    """One round of proposals against the same parent model.

    Two ways to draw it. Without `--analysis` it is one request for N samples,
    which is the cheaper draw. With it, each agent gets its own conversation and
    the `run_python` tool over the training days, which costs N prefills and
    buys proposals written by agents that have actually looked at the data.
    """
    llm_model = settings.llm_model or served(settings.url)
    decode = Decode(
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        max_tokens=settings.max_tokens,
        thinking_token_budget=settings.thinking_token_budget,
    )
    shared = {
        "url": settings.url,
        "llm_model": llm_model,
        "api_key": settings.api_key,
        "agents": settings.agents,
        "decode": decode,
    }
    print(f"asking {settings.agents} agents on {llm_model} for a generating process")

    if not settings.analysis:
        async with IIDSearch(**shared, timeout=settings.search_timeout) as search:
            proposals = await search.propose(start, feedback=brief)
    else:
        pool = Analysis(
            setup=setup.__name__,
            description=setup.DESCRIPTION,
            workers=settings.analysis_workers,
            timeout=settings.analysis_timeout,
            setup_env={
                "BIRTHDATES_DATA": settings.data,
                "BIRTHDATES_HOLDOUT_FROM": str(settings.holdout_from),
            },
        )
        print(f"  with {settings.analysis_workers} analysis workers, {settings.tool_budget} calls each")
        async with pool, AgentSearch(**shared, analysis=pool, tool_budget=settings.tool_budget) as search:
            proposals = await search.propose(start, feedback=brief)

    for proposal in proposals:
        if not proposal.ok:
            print(f"  agent {proposal.agent}: {type(proposal.error).__name__}: {proposal.error}")
    kept = models(proposals)
    print(f"{len(kept)}/{len(proposals)} proposals parsed")
    if failed := failures(proposals, settings.out):
        print(f"failed agents: {failed}")
    if talk := transcripts(proposals, settings.out):
        print(f"transcripts: {talk}")
    return kept


def failures(proposals, out: Path) -> Path | None:
    """Write out what the failed agents had done, if anything.

    A failed proposal is never committed to its transcript, so without this the
    six snippets an agent ran and every number it read vanish with the model it
    did not write -- and the report only keeps candidates that were fitted. The
    error carries the exchange precisely so somebody can read it afterwards, and
    a run where most agents failed is a run where that is the only evidence
    there is.
    """
    records = [
        {
            "agent": p.agent,
            "error": f"{type(p.error).__name__}: {p.error}",
            "reply": getattr(p.error, "reply", None),
            "finish_reason": getattr(p.error, "finish_reason", None),
            "exchange": getattr(p.error, "exchange", []),
        }
        for p in proposals
        if not p.ok
    ]
    if not records:
        return None
    out.mkdir(parents=True, exist_ok=True)
    path = out / "failures.jsonl"
    with open(path, "w") as handle:
        for record in records:
            json.dump(record, handle, default=str)
            handle.write("\n")
    return path


def transcripts(proposals, out: Path) -> Path | None:
    """Write down every conversation the round had, successes included.

    `failures.jsonl` keeps the exchange behind an agent that wrote nothing, and
    the report keeps the model an agent that succeeded wrote -- but not the
    snippets it ran to get there. For a tool-using agent that is most of what
    happened, and it is the only account of why a candidate looks the way it
    does, so it is written down rather than dropped when the round returns.

    `candidate` is the index the kept proposal takes in the fit, since the run
    fits the starting model first and the proposals after it in order; that is
    what joins a conversation to its row in the report.
    """
    kept = 0
    records = []
    for proposal in proposals:
        if proposal.ok:
            kept += 1
        records.append(
            {
                "agent": proposal.agent,
                "ok": proposal.ok,
                "candidate": kept if proposal.ok else None,
                "error": None if proposal.ok else f"{type(proposal.error).__name__}: {proposal.error}",
                "transcript": list(proposal.transcript),
            }
        )
    if not records:
        return None
    out.mkdir(parents=True, exist_ok=True)
    path = out / "transcripts.jsonl"
    with open(path, "w") as handle:
        for record in records:
            json.dump(record, handle, default=str)
            handle.write("\n")
    return path


def table(evaluations) -> str:
    """The run, ranked by held-out likelihood."""
    columns = (
        ("elpd/day", "posterior.elpd_holdout_per_day", "{:>9.3f}"),
        ("cover95", "posterior.coverage95", "{:>7.3f}"),
        ("kl_min", "posterior.kl_min", "{:>7.2f}"),
        ("unid", "posterior.unidentified", "{:>4.0f}"),
        ("rhat", "posterior.rhat_max", "{:>5.2f}"),
        ("ess_tail", "posterior.ess_tail_min", "{:>8.0f}"),
        ("div", "posterior.divergences", "{:>5.0f}"),
        ("secs", "posterior.mcmc_seconds", "{:>6.1f}"),
        ("pars", "posterior.parameters", "{:>4.0f}"),
    )
    order = sorted(
        evaluations,
        key=lambda e: (not e.ok, -e.scores.get(RANK, float("-inf"))),
    )
    lines = ["  #  " + " ".join(f"{name:>{len(fmt.format(0)) }}" for name, _, fmt in columns)]
    for evaluation in order:
        label = f"{evaluation.index:>3}" + ("*" if evaluation.index == 0 else " ")
        if not evaluation.ok:
            reason = "; ".join(f"{type(e).__name__}: {e}" for e in evaluation.errors.values())
            lines.append(f"{label} {reason.splitlines()[-1][:100]}")
            continue
        cells = (fmt.format(evaluation.scores[key]) if key in evaluation.scores else "-"
                 for _, key, fmt in columns)
        lines.append(f"{label} " + " ".join(cells))
    return "\n".join(lines)


async def main(argv: list[str] | None = None) -> int:
    settings = options(argv)
    if not settings.no_search and not settings.url:
        raise SystemExit("--url (or $DISCOVERY_URL) is required unless --no-search")
    # The analysis pool says nothing unless the queue makes agents wait, which
    # is the one thing worth seeing while a round runs.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    train, holdout = data.load(settings.data, settings.holdout_from)
    print(f"train   {len(train):>5} days, {train.span}")
    print(f"holdout {len(holdout):>5} days, {holdout.span}")

    start = Model(SOURCE)
    brief = BRIEF.format(
        train=len(train),
        train_span=train.span,
        holdout=len(holdout),
        holdout_span=holdout.span,
        mean=sum(train.births) / len(train),
        chains=settings.chains,
        warmup=settings.warmup,
        samples=settings.samples,
    )

    candidates = (start,)
    if not settings.no_search:
        candidates += await propose(settings, start, brief)

    budget = Budget(
        data=settings.data,
        holdout_from=settings.holdout_from,
        warmup=settings.warmup,
        samples=settings.samples,
        chains=settings.chains,
        seed=settings.seed,
        max_parameters=settings.max_parameters,
    )
    print(f"fitting {len(candidates)} candidates, {settings.concurrency} at a time")
    evaluations = await evaluate(
        candidates,
        [Posterior(budget=budget, timeout=settings.fit_timeout)],
        concurrency=settings.concurrency,
        timeout=settings.fit_timeout + 60.0,
    )

    report = write(evaluations, settings.out / "report.jsonl")
    print(f"\n{table(evaluations)}\n\n* the model the search started from")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
