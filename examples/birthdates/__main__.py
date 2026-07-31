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
import os
import sys
import urllib.request
from pathlib import Path

from discovery.evaluator import evaluate
from discovery.model import Model
from discovery.report import write
from discovery.search import Decode, IIDSearch, models

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
    """One i.i.d. round: N proposals against the same parent, in one request."""
    llm_model = settings.llm_model or served(settings.url)
    print(f"asking {settings.agents} agents on {llm_model} for a generating process")
    async with IIDSearch(
        url=settings.url,
        llm_model=llm_model,
        api_key=settings.api_key,
        agents=settings.agents,
        timeout=settings.search_timeout,
        decode=Decode(
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            max_tokens=settings.max_tokens,
            thinking_token_budget=settings.thinking_token_budget,
        ),
    ) as search:
        proposals = await search.propose(start, feedback=brief)
    for proposal in proposals:
        if not proposal.ok:
            print(f"  agent {proposal.agent}: {type(proposal.error).__name__}: {proposal.error}")
    kept = models(proposals)
    print(f"{len(kept)}/{len(proposals)} proposals parsed")
    return kept


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
