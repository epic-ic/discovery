"""Search for the map from parameters to malaria age curves.

    uv run python -m examples.malaria --url http://host:8000/v1

msinf's observational model is held fixed -- the study random effects, the
incidence and prevalence likelihoods, the site EIR priors, `ft_study`, the age
grid, the demography, the age-bin averaging -- and what varies between
candidates is only the curve provider: the map from parameters to prevalence and
incidence over age. One round: ask N agents for a curve family, fit every
proposal and the model it started from under both splits with the same NUTS
budget, score them all, write the report.

Two starting models, and `--start both` is the point of the exercise. `scratch`
is deliberately poor and asks what a search finds from nothing; `griffin` is the
incumbent and asks whether a search seeded with a misspecified mechanism corrects
it. Whichever wins, expect something flexible in age x EIR with no mechanism in
it to be near the top: that is not a failure, it is the empirical ceiling on
held-out likelihood, and it is the number the mechanistic candidates should be
judged against.

`--no-search` skips the language model and fits only the starting models, which
is the fastest way to see the whole evaluation path work -- though "fastest" here
is a fit per split per model, at tens of minutes each.

`--analysis` gives every agent the training rows before it proposes: the study
table with the folds this run scores removed, the model's age grid and the
demography that weights it. numpy and pandas, no jax, and no candidate is run --
the gates already do that, in the fit, where a backend is already paid for.
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

from examples.malaria import analysis as setup
from examples.malaria import data
from examples.malaria.fit import Budget, Posterior
from examples.malaria.initial import BRIEF, STARTS

#: the score models are ranked by. The age-band split rather than the study
#: split: it leaves each site's `log_eir` informed by that study's training rows,
#: so what it measures is age structure, which is what a model of acquired
#: immunity actually claims to explain. The study split is blunter -- a held-out
#: study's EIR sits on its prior, and that variance can swamp the difference
#: between two reasonable candidates -- so it is reported, not ranked on.
RANK = "posterior.age_band.elpd_holdout"

#: roughly what one fit of this model costs, in minutes, at the default budget.
#: Load-bearing: it sizes `--fit-timeout` and it is what the proposer is told,
#: so a candidate that is much more expensive to solve gets killed rather than
#: silently taking the run's whole afternoon.
FIT_MINUTES = 20.0

#: how much longer than `FIT_MINUTES x splits` a candidate may take before it is
#: killed. Compilation, the ODE-bound scoring pass and a curve family slower than
#: the incumbent's all live in this factor.
HEADROOM = 3.0


def options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="examples.malaria", description=__doc__)
    llm = parser.add_argument_group("inference server")
    llm.add_argument(
        "--url",
        default=os.environ.get("DISCOVERY_URL"),
        help="OpenAI-compatible base url, e.g. http://host:8000/v1 [$DISCOVERY_URL]",
    )
    llm.add_argument("--llm-model", default=None, help="defaults to whatever the server serves")
    llm.add_argument("--api-key", default=os.environ.get("DISCOVERY_API_KEY", "EMPTY"))
    llm.add_argument("--agents", type=int, default=8, help="proposals drawn per starting model")
    llm.add_argument("--temperature", type=float, default=1.0)
    llm.add_argument("--top-p", type=float, default=0.95)
    llm.add_argument("--top-k", type=int, default=50)
    llm.add_argument("--max-tokens", type=int, default=8192)
    llm.add_argument(
        "--thinking-token-budget",
        type=int,
        default=1024,
        help="bound the reasoning; left alone it can spend max-tokens and return nothing",
    )
    llm.add_argument("--search-timeout", type=float, default=1800.0)
    llm.add_argument("--no-search", action="store_true", help="fit only the starting models")

    tools = parser.add_argument_group("data analysis at proposal time")
    tools.add_argument(
        "--analysis",
        action="store_true",
        help="let agents explore the training rows before proposing; needs a server started "
        "with --enable-auto-tool-choice and --tool-call-parser",
    )
    tools.add_argument("--analysis-workers", type=int, default=4, help="snippets running at once")
    tools.add_argument("--tool-budget", type=int, default=6, help="snippets one agent may run")
    tools.add_argument("--analysis-timeout", type=float, default=20.0, help="seconds per snippet")

    start = parser.add_argument_group("what the search starts from")
    start.add_argument(
        "--start",
        choices=(*STARTS, "both"),
        default="both",
        help="scratch: no immunity, no age structure. griffin: the incumbent, "
        "for correction rather than discovery. both: one search from each, "
        "sharing scaffolding, budget and scores exactly",
    )

    fit = parser.add_argument_group("fit, identical for every candidate")
    fit.add_argument("--table", default=str(data.DEFAULT_TABLE), help="the <=20y study table")
    fit.add_argument("--eir-priors", default=str(data.DEFAULT_EIR_PRIORS))
    fit.add_argument("--mortality", default=data.DEFAULT_MORTALITY,
                     help="per-site WPP death rates; omitted keeps dmeq's uniform rate")
    fit.add_argument("--splits", default=",".join(data.SPLITS),
                     help="comma separated; one fit each, reported separately")
    fit.add_argument("--holdout-seed", type=int, default=42, help="the study-split fold")
    fit.add_argument("--n-studies", type=int, default=3, help="studies held out by the study split")
    fit.add_argument("--from-age", type=float, default=5.0,
                     help="first held-out age band; must fall between age brackets")
    fit.add_argument("--warmup", type=int, default=400)
    fit.add_argument("--samples", type=int, default=400, help="post-warmup draws per chain")
    fit.add_argument("--chains", type=int, default=4)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--max-parameters", type=int, default=200)
    fit.add_argument("--concurrency", type=int, default=1,
                     help="fits in flight at once; each is chains x one thread and a lot of memory")
    fit.add_argument("--fit-timeout", type=float, default=None,
                     help=f"seconds; defaults to {FIT_MINUTES:.0f}min x splits x {HEADROOM:g}")

    parser.add_argument("--out", type=Path, default=Path("runs/malaria"))
    return parser.parse_args(argv)


def served(url: str) -> str:
    """The first model the server offers, so the url is the only thing to know."""
    with urllib.request.urlopen(f"{url.rstrip('/')}/models", timeout=30) as response:
        offered = json.load(response)["data"]
    if not offered:
        raise SystemExit(f"{url} serves no models")
    return offered[0]["id"]


async def propose(settings: argparse.Namespace, start: Model, brief: str, pool: Analysis | None = None):
    """One round of proposals against one starting model.

    Without `pool`, an i.i.d. draw: one request for N samples, every one of them
    rewriting the same parent. With it, N agents in N conversations, each able to
    read the training rows before it writes anything -- the same draw, paid for
    differently, and the reason to pay is that a curve family proposed by
    somebody who has looked at how prevalence moves with age and EIR is a better
    draw than one proposed blind.

    `--start both` runs this twice, and the two rounds do not see each other --
    the comparison is between what a search finds from nothing and what it finds
    from the incumbent, and it stops being that comparison the moment either
    round can read the other's proposals. The pool is shared across the two
    rounds and carries nothing between them: every snippet gets a fresh
    namespace, whoever asked for it.
    """
    decode = Decode(
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        max_tokens=settings.max_tokens,
        thinking_token_budget=settings.thinking_token_budget,
    )
    shared = {
        "url": settings.url,
        "llm_model": settings.llm_model,
        "api_key": settings.api_key,
        "agents": settings.agents,
        "decode": decode,
    }
    if pool is None:
        search = IIDSearch(**shared, timeout=settings.search_timeout)
    else:
        search = AgentSearch(**shared, analysis=pool, tool_budget=settings.tool_budget)
    async with search:
        proposals = await search.propose(start, feedback=brief)

    for proposal in proposals:
        if not proposal.ok:
            print(f"  agent {proposal.agent}: {type(proposal.error).__name__}: {proposal.error}")
    kept = models(proposals)
    print(f"  {len(kept)}/{len(proposals)} proposals parsed")
    return kept, proposals


def failures(proposals, out: Path) -> Path | None:
    """Write out what the failed agents had said, if anything.

    A failed proposal is never committed to its transcript, and the report only
    keeps candidates that were fitted, so without this a run where most agents
    failed leaves no evidence of why. Both rounds go to one file, each record
    naming the model it was proposing against.
    """
    records = [
        {
            "start": parent,
            "agent": p.agent,
            "error": f"{type(p.error).__name__}: {p.error}",
            "reply": getattr(p.error, "reply", None),
            "finish_reason": getattr(p.error, "finish_reason", None),
            "exchange": getattr(p.error, "exchange", []),
        }
        for parent, batch in proposals
        for p in batch
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


def table(evaluations, origin: dict[int, str], starts: set[int], splits: tuple[str, ...]) -> str:
    """The run, ranked by held-out likelihood on the age-band split.

    Both splits' `elpd_holdout` are shown and neither is averaged into the other:
    they answer different questions and the study split is much the noisier.
    """
    columns = [("from", None, "{:>8}")]
    columns += [(f"elpd/{kind[:5]}", f"posterior.{kind}.elpd_holdout", "{:>11.1f}") for kind in splits]
    columns += [
        ("cover90", f"posterior.{splits[-1]}.coverage90_holdout_prev_m", "{:>7.3f}"),
        ("kl_min", "posterior.kl_min", "{:>7.2f}"),
        ("unid", "posterior.unidentified", "{:>4.0f}"),
        ("rhat", "posterior.rhat_max", "{:>5.2f}"),
        ("ess_tail", "posterior.ess_tail_min", "{:>8.0f}"),
        ("div", "posterior.divergences", "{:>5.0f}"),
        ("mins", "posterior.mcmc_seconds", "{:>6.1f}"),
        ("pars", "posterior.parameters", "{:>4.0f}"),
    ]
    order = sorted(evaluations, key=lambda e: (not e.ok, -e.scores.get(RANK, float("-inf"))))

    lines = ["  #  " + " ".join(f"{name:>{len(fmt.format(0)) }}" for name, _, fmt in columns)]
    for evaluation in order:
        label = f"{evaluation.index:>3}" + ("*" if evaluation.index in starts else " ")
        parent = origin.get(evaluation.index, "?")
        if not evaluation.ok:
            reason = "; ".join(f"{type(e).__name__}: {e}" for e in evaluation.errors.values())
            lines.append(f"{label} {parent:>8} {reason.splitlines()[-1][:96]}")
            continue
        cells = []
        for name, key, fmt in columns:
            if key is None:
                cells.append(fmt.format(parent))
            elif key not in evaluation.scores:
                cells.append("-".rjust(len(fmt.format(0))))
            else:
                value = evaluation.scores[key]
                cells.append(fmt.format(value / 60.0 if name == "mins" else value))
        lines.append(f"{label} " + " ".join(cells))
    return "\n".join(lines)


async def main(argv: list[str] | None = None) -> int:
    settings = options(argv)
    if not settings.no_search and not settings.url:
        raise SystemExit("--url (or $DISCOVERY_URL) is required unless --no-search")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    splits = tuple(s.strip() for s in settings.splits.split(",") if s.strip())
    if unknown := set(splits) - set(data.SPLITS):
        raise SystemExit(f"unknown split(s) {sorted(unknown)}; expected some of {list(data.SPLITS)}")

    # Fail here rather than inside a child process: without the tables there is
    # nothing to run, and it is worth saying so once, up front.
    inputs = data.inputs(settings.table, settings.eir_priors, settings.mortality)
    summary = data.read(inputs.table)
    print(f"table   {summary}")
    for kind in splits:
        name = data.label(kind, settings.holdout_seed, settings.n_studies, settings.from_age)
        print(f"split   {name}")

    timeout = settings.fit_timeout
    if timeout is None:
        timeout = FIT_MINUTES * 60.0 * len(splits) * HEADROOM
    brief = BRIEF.format(
        table=summary,
        study=settings.n_studies,
        from_age=settings.from_age,
        chains=settings.chains,
        warmup=settings.warmup,
        samples=settings.samples,
        minutes=FIT_MINUTES,
        inc_max=Budget.inc_max,
    )

    # One pool for both rounds, booted before either of them so the first agent
    # does not pay for the import, and closed once both are done. It is given
    # the same table and the same fold the fit will use, and hides what that
    # fold holds out -- see `analysis.py`.
    pool = None
    if settings.analysis and not settings.no_search:
        pool = Analysis(
            setup=setup.__name__,
            description=setup.DESCRIPTION,
            workers=settings.analysis_workers,
            timeout=settings.analysis_timeout,
            setup_env={
                "MALARIA_TABLE": str(inputs.table),
                "MALARIA_EIR_PRIORS": str(inputs.eir_priors),
                "MALARIA_MORTALITY": inputs.mortality or "",
                "MALARIA_SPLITS": ",".join(splits),
                "MALARIA_HOLDOUT_SEED": str(settings.holdout_seed),
                "MALARIA_N_STUDIES": str(settings.n_studies),
                "MALARIA_FROM_AGE": str(settings.from_age),
            },
        )
        await pool.start()
        print(f"        {settings.analysis_workers} analysis workers, {settings.tool_budget} calls each")

    chosen = tuple(STARTS) if settings.start == "both" else (settings.start,)
    candidates: list[Model] = []
    #: every candidate's parent, so a row says which search found it. The
    #: starting models are their own parents, and are the rows marked `*`.
    origin: dict[int, str] = {}
    starts: set[int] = set()
    drawn: list[tuple[str, tuple]] = []
    try:
        for name in chosen:
            starts.add(len(candidates))
            origin[len(candidates)] = name
            candidates.append(Model(STARTS[name]))
            if settings.no_search:
                continue
            settings.llm_model = settings.llm_model or served(settings.url)
            print(f"asking {settings.agents} agents on {settings.llm_model} for a curve family, from {name}")
            kept, proposals = await propose(settings, Model(STARTS[name]), brief, pool)
            drawn.append((name, proposals))
            for model in kept:
                origin[len(candidates)] = name
                candidates.append(model)
    finally:
        # Before the fits rather than after them: the workers hold the table and
        # a few hundred megabytes each, and the fits want every core.
        if pool is not None:
            await pool.aclose()
    if failed := failures(drawn, settings.out):
        print(f"failed agents: {failed}")

    budget = Budget(
        table=str(inputs.table),
        eir_priors=str(inputs.eir_priors),
        mortality=inputs.mortality,
        splits=splits,
        holdout_seed=settings.holdout_seed,
        n_studies=settings.n_studies,
        from_age=settings.from_age,
        warmup=settings.warmup,
        samples=settings.samples,
        chains=settings.chains,
        seed=settings.seed,
        max_parameters=settings.max_parameters,
    )
    print(
        f"fitting {len(candidates)} candidates x {len(splits)} splits, "
        f"{settings.concurrency} at a time, {timeout / 60:.0f}min each"
    )
    evaluations = await evaluate(
        candidates,
        [Posterior(budget=budget, timeout=timeout)],
        concurrency=settings.concurrency,
        timeout=timeout + 120.0,
    )

    report = write(evaluations, settings.out / "report.jsonl")
    print(f"\n{table(evaluations, origin, starts, splits)}\n\n* the models the search started from")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
