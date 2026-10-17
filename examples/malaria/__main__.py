"""Search for the map from parameters to malaria age curves.

    uv run python -m examples.malaria --url http://host:8000/v1

msinf's observational model is held fixed -- the study random effects, the
incidence and prevalence likelihoods, the site EIR priors, `ft_study`, the age
grid, the demography, the age-bin averaging -- and what varies between
candidates is only the mechanism: the map from parameters to prevalence and
incidence over age. One round: ask N agents for a block, fit every proposal and
the model it started from under the same NUTS budget, score them all, write the
report.

`--target` says how much of the mechanism is on the table. It is one model with two
apertures, not two models: `curves` opens the immunity model, the equilibrium state
solve and the driver; `immunity` opens the `eps -> (foi, phi, q, b)` block alone and
leaves the rest below it as fixed but readable text. The incumbent samples the same
18 parameters either way, so its number is the same number at both targets, and a
`curves` run is the empirical ceiling an `immunity` run's mechanistic candidates
should be judged against.

`--start` picks the one model the round rewrites. `scratch` is deliberately poor
and asks what a search finds from nothing; `griffin` is the incumbent and asks
whether a search seeded with a misspecified mechanism corrects it. One per run:
comparing them means running both and reading the two reports, which the shell
does better than an option here -- a single run that drew from both would double
its own length and still keep the two rounds blind to each other, since they stop
being that comparison the moment either can read the other's proposals.

At `--target curves`, expect something flexible in age x EIR with no mechanism in
it to be near the top: that is not a failure, it is the empirical ceiling on
held-out likelihood.

The two halves of a run can be separated. `--propose-only` draws the round and
writes every candidate to `<out>/proposals` as source, stopping before anything
is fitted; `--proposals <dir>` fits a round written that way and never opens a
connection. The server is then needed only for the minutes the proposals take,
not the hours the fits do, and the same round can be refitted later -- under a
longer budget, or after the evaluation changes -- without redrawing it. A drawn
round also leaves `<out>/transcripts.jsonl`: every agent's conversation, the
snippets it ran included, which is the only account of why a candidate looks the
way it does once the proposals are just source files.

`--no-search` skips the language model and fits only the starting models, which
is the fastest way to see the whole evaluation path work -- though "fastest" here
is a fit per split per model, at tens of minutes each.

`--analysis` gives every agent the training rows before it proposes: the study
table with the fold this run scores removed, the model's age grid and the
demography that weights it. numpy and pandas, no jax, and no candidate is run --
the gates already do that, in the fit, where a backend is already paid for.
"""

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

from discovery.analysis import Analysis
from discovery.evaluator import evaluate
from discovery.model import Model
from discovery.report import write
from discovery.search import AgentSearch, Decode, IIDSearch, models

from examples.estimators import elpd_diff
from examples.malaria import analysis as setup
from examples.malaria import data
from examples.malaria.fit import Budget, Posterior
from examples.malaria.initial import BRIEF, START_NAMES, STARTS, TARGETS

#: the score models are ranked by, and the one number this example has to rank
#: on -- see `data.SPLITS` for why the age-band split it used to rank on is gone.
#:
#: Read it knowing what it is blunt about. A held-out study's `log_eir` sits on
#: its prior, so a large part of this number is the EIR-to-outcome mapping
#: integrated over a wide prior: machinery every candidate shares, which adds a
#: common offset and a lot of variance while discriminating weakly between two
#: reasonable curve families. It separates working candidates from broken ones
#: cleanly; it does not, on 35 scoring units, reliably order the top few. Until
#: the per-unit log-likelihoods are reported and differences carry a standard
#: error, treat the top of the table as unordered.
RANK = "posterior.study.elpd_holdout"

#: roughly what one fit of this model costs, in minutes, at the default budget.
#: Load-bearing: it sizes `--fit-timeout` and it is what the proposer is told,
#: so a candidate that is much more expensive to solve gets killed rather than
#: silently taking the run's whole afternoon.
#:
#: Measured, not guessed: Griffin at 150 + 150 x 4 chains took 136 minutes of
#: MCMC on a 20-core laptop that was doing other work (runs/beat/r00). A
#: dedicated cluster core is faster, which only makes the timeout generous.
FIT_MINUTES = 140.0

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
    llm.add_argument(
        "--propose-only",
        action="store_true",
        help="draw the round, write the candidates to <out>/proposals, and stop before "
        "any fitting -- so the inference server is only needed for the cheap half",
    )
    llm.add_argument(
        "--proposals",
        type=Path,
        default=None,
        help="fit a round written earlier by --propose-only, instead of drawing one; "
        "needs no server, and --target/--start are read from the saved round",
    )

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
        "--target",
        choices=TARGETS,
        default="curves",
        help="curves: the whole solve, parameters -> (prev_m, inc) over age. "
        "immunity: the eps -> (foi, phi, q, b) block alone, with the equilibrium "
        "state solve and the demography fixed",
    )
    start.add_argument(
        "--start",
        choices=START_NAMES,
        default="griffin",
        help="scratch: deliberately poor -- no immunity and no age structure at "
        "the curves target, immunity that does not accumulate at the immunity "
        "one. griffin: the incumbent, for correction rather than discovery. "
        "One per run: the two are compared by running both and reading the two "
        "reports, not by mixing them into one",
    )

    fit = parser.add_argument_group("fit, identical for every candidate")
    fit.add_argument("--table", default=str(data.DEFAULT_TABLE), help="the <=20y study table")
    fit.add_argument("--eir-priors", default=str(data.DEFAULT_EIR_PRIORS))
    fit.add_argument(
        "--mortality",
        default=data.DEFAULT_MORTALITY,
        help="per-site WPP death rates; omitted keeps dmeq's uniform rate",
    )
    fit.add_argument(
        "--splits", default=",".join(data.SPLITS), help="comma separated; one fit each, reported separately"
    )
    fit.add_argument("--holdout-seed", type=int, default=42, help="the study-split fold")
    fit.add_argument("--n-studies", type=int, default=3, help="studies held out by the study split")
    fit.add_argument("--warmup", type=int, default=Budget.warmup, help="warmup draws per chain")
    fit.add_argument("--samples", type=int, default=Budget.samples, help="post-warmup draws per chain")
    fit.add_argument("--chains", type=int, default=Budget.chains)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--max-parameters", type=int, default=200)
    fit.add_argument(
        "--inc-max",
        type=float,
        default=20.0,
        help="clinical episodes per person-year no age class may exceed on a prior "
        "draw. `inf` fits whatever the prior permits -- negative incidence still "
        "fails -- which is how to ask what a candidate the gate rejected would have "
        "scored, at the price of the fit it was there to save",
    )
    fit.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="fits in flight at once; each is chains x one thread and a lot of memory",
    )
    fit.add_argument(
        "--fit-timeout",
        type=float,
        default=None,
        help=f"seconds; defaults to {FIT_MINUTES:.0f}min x splits x {HEADROOM:g}",
    )

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

    One round per run, against one parent. Comparing the two starts means two
    runs: the comparison is between what a search finds from nothing and what it
    finds from the incumbent, and it stops being that comparison the moment either
    round can read the other's proposals -- so there is nothing for a single run to
    share between them except the fits, which are the expensive part and not the
    part that would be saved.
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


def transcripts(proposals, placed: dict[tuple[str, int], int], out: Path) -> Path | None:
    """Write down every conversation the round had, successes included.

    `failures.jsonl` keeps the exchange behind an agent that never produced a
    model, and the proposals directory keeps the model an agent that did
    produced -- but not how it got there. For a tool-using agent that is most of
    what happened: the snippets it ran against the data, what came back, and the
    reasoning it wrote off the back of them. It is the only record of why a
    candidate looks the way it does, and it is gone as soon as the round
    returns, so it is written beside the round rather than reconstructed later.

    One record per agent, naming the model it proposed against and, when the
    proposal was kept, the candidate index it became -- which is what joins a
    conversation to its row in the report and its file in `proposals/`.
    """
    records = [
        {
            "start": parent,
            "agent": p.agent,
            "ok": p.ok,
            "candidate": placed.get((parent, p.agent)),
            "error": None if p.ok else f"{type(p.error).__name__}: {p.error}",
            "transcript": list(p.transcript),
        }
        for parent, batch in proposals
        for p in batch
    ]
    if not records:
        return None
    out.mkdir(parents=True, exist_ok=True)
    path = out / "transcripts.jsonl"
    with open(path, "w") as handle:
        for record in records:
            json.dump(record, handle, default=str)
            handle.write("\n")
    return path


def save(candidates: list[Model], origin: dict[int, str], settings, out: Path) -> Path:
    """Write a drawn round to disk as source files, one per candidate.

    The half of a run that needs the language model is minutes of it; the half
    that needs a machine is hours. Writing the round down between them means the
    server has to be up only for the first, and that the same round can be
    refitted later -- under a longer budget, or after the evaluation changes --
    without paying for new proposals or wondering whether they are the same ones.

    Source files rather than one json blob because a candidate *is* a source
    file: it can be read, diffed and edited where it lands, and the index and
    parent a fit needs are in the name. `0000` is the model the round started
    from, exactly as it was proposed against.
    """
    directory = out / "proposals"
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.py"):
        old.unlink()
    for index, model in enumerate(candidates):
        (directory / f"{index:04d}-{origin[index]}.py").write_text(model.source)
    (directory / "round.json").write_text(
        json.dumps(
            {
                "target": settings.target,
                "start": settings.start,
                "llm_model": settings.llm_model,
                "agents": settings.agents,
                "analysis": settings.analysis,
                "tool_budget": settings.tool_budget,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "top_k": settings.top_k,
                "max_tokens": settings.max_tokens,
                "thinking_token_budget": settings.thinking_token_budget,
                "candidates": len(candidates),
            },
            indent=2,
        )
        + "\n"
    )
    return directory


def load(directory: Path) -> tuple[list[Model], dict[int, str], dict]:
    """Read back a round written by `save`: the candidates and their parents.

    The starting model comes back from the file too rather than from `STARTS`,
    so a round fitted later is fitted against the same parent it was drawn
    against even if the starting models have since been edited.
    """
    paths = sorted(directory.glob("*.py"))
    if not paths:
        raise SystemExit(f"no candidates in {directory}")
    candidates = [Model(path.read_text()) for path in paths]
    origin = {index: path.stem.split("-", 1)[-1] for index, path in enumerate(paths)}
    meta_path = directory / "round.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return candidates, origin, meta


#: the split the table's `vs best` column pairs on, taken from `RANK` so the
#: two can never disagree about which fold is being compared.
RANKED_SPLIT = RANK.split(".")[1]


def pointwise(scores, kind: str, fold: str = "holdout") -> tuple[float, ...] | None:
    """One candidate's per-unit elpd on `fold` under `kind`, cells then rows.

    `None` if it was not scored -- a gated or failed candidate has no vector,
    and neither does a report written before `pointwise_*` existed.
    """
    parts = []
    for stream in ("incidence", "prevalence"):
        vector = scores.get(f"posterior.{kind}.pointwise_{fold}_{stream}")
        if vector is None:
            return None
        parts.extend(float(x) for x in vector)
    return tuple(parts)


def leads(order, kind: str) -> dict[int, str]:
    """Each candidate's lead over the best one, as `diff+-se`, by index.

    The column that stops the table being read as a ranking when it is not one.
    The totals in `elpd/study` differ by tens of nats on a fold of a few dozen
    units, and without the paired error beside them there is no way to see that
    the top few are indistinguishable. Empty for the reference row itself, and
    for anything that cannot be paired with it.
    """
    scored = [e for e in order if e.ok and pointwise(e.scores, kind) is not None]
    if not scored:
        return {}
    best, *rest = scored
    reference = pointwise(best.scores, kind)
    out = {best.index: "ref"}
    for evaluation in rest:
        try:
            difference, se = elpd_diff(pointwise(evaluation.scores, kind), reference)
        except ValueError:
            # Different fold, so not comparable. It should not happen in one
            # run -- every candidate is fitted under the same split -- and if it
            # does, saying nothing beats printing a number that means nothing.
            continue
        out[evaluation.index] = f"{difference:+.0f}+-{se:.0f}" if se == se else f"{difference:+.0f}"
    return out


def table(evaluations, origin: dict[int, str], starts: set[int], splits: tuple[str, ...]) -> str:
    """The run, ranked by held-out likelihood on the study split -- see `RANK`.

    One `elpd_holdout` column per split fitted, and they are never averaged into
    each other: different splits answer different questions.

    `vs best` is the paired difference against the top row and its standard
    error, both in nats. Read it before the `elpd` column it qualifies: a lead
    smaller than about twice its error is not a lead, and on this fold that
    covers most of the gap between reasonable candidates.
    """
    columns = [("from", None, "{:>8}")]
    columns += [(f"elpd/{kind[:5]}", f"posterior.{kind}.elpd_holdout", "{:>11.1f}") for kind in splits]
    columns += [
        ("vs best", "", "{:>11}"),
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
    against_best = leads(order, RANKED_SPLIT)

    lines = ["  #  " + " ".join(f"{name:>{len(fmt.format(0))}}" for name, _, fmt in columns)]
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
            elif key == "":
                cells.append(fmt.format(against_best.get(evaluation.index, "-")))
            elif key not in evaluation.scores:
                cells.append("-".rjust(len(fmt.format(0))))
            else:
                value = evaluation.scores[key]
                cells.append(fmt.format(value / 60.0 if name == "mins" else value))
        lines.append(f"{label} " + " ".join(cells))
    return "\n".join(lines)


def fitted(settings: argparse.Namespace, splits: tuple[str, ...], budget: Budget) -> Path:
    """Write down how the round beside it was fitted, as `fit.json`.

    The report says what each candidate scored and not under what, and two rounds
    are comparable only under the same split and budget. `feedback` reads the
    split before it shows an agent anything -- another fold's training rows are
    this fold's held-out ones -- and `rounds` groups its table by both.
    """
    path = settings.out / "fit.json"
    path.write_text(
        json.dumps(
            {
                "split": data.label(splits[0], settings.holdout_seed, settings.n_studies),
                "splits": list(splits),
                "holdout_seed": settings.holdout_seed,
                "n_studies": settings.n_studies,
                "warmup": budget.warmup,
                "samples": budget.samples,
                "chains": budget.chains,
                "seed": budget.seed,
                "inc_max": budget.inc_max,
                "host": socket.gethostname(),
                "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
        + "\n"
    )
    return path


async def main(argv: list[str] | None = None) -> int:
    settings = options(argv)
    #: whether this run talks to the inference server at all. A round is either
    #: drawn now, read back from one drawn earlier, or skipped entirely.
    searching = not settings.no_search and settings.proposals is None
    if settings.propose_only and not searching:
        raise SystemExit("--propose-only draws a round, so it cannot be used with --no-search or --proposals")
    if searching and not settings.url:
        raise SystemExit("--url (or $DISCOVERY_URL) is required unless --no-search or --proposals")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    splits = tuple(s.strip() for s in settings.splits.split(",") if s.strip())
    if unknown := set(splits) - set(data.SPLITS):
        raise SystemExit(f"unknown split(s) {sorted(unknown)}; expected some of {list(data.SPLITS)}")

    # Fail here rather than inside a child process: without the tables there is
    # nothing to run, and it is worth saying so once, up front.
    inputs = data.inputs(settings.table, settings.eir_priors, settings.mortality)
    summary = data.read(inputs.table)
    if settings.proposals is None:
        print(f"target  {settings.target}")
    print(f"table   {summary}")
    for kind in splits:
        print(f"split   {data.label(kind, settings.holdout_seed, settings.n_studies)}")

    timeout = settings.fit_timeout
    if timeout is None:
        timeout = FIT_MINUTES * 60.0 * len(splits) * HEADROOM
    brief = BRIEF.format(
        table=summary,
        study=settings.n_studies,
        chains=settings.chains,
        warmup=settings.warmup,
        samples=settings.samples,
        minutes=FIT_MINUTES,
        inc_max=Budget.inc_max,
    )

    # Booted before the round so the first agent does not pay for the import, and
    # closed before the fits, which want every core. It is given the same table
    # and the same fold the fit will use, and hides what that fold holds out --
    # see `analysis.py`.
    pool = None
    if settings.analysis and searching:
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
            },
        )
        await pool.start()
        print(f"        {settings.analysis_workers} analysis workers, {settings.tool_budget} calls each")

    name = settings.start
    #: every candidate's parent, so a row says which search it came from. The
    #: starting model is its own parent, and is the row marked `*`.
    candidates: list[Model] = []
    origin: dict[int, str] = {}
    starts: set[int] = {0}
    drawn: list[tuple[str, tuple]] = []
    #: which candidate each kept proposal became, by the agent that wrote it
    placed: dict[tuple[str, int], int] = {}
    if settings.proposals is not None:
        candidates, origin, meta = load(settings.proposals)
        name = origin.get(0, name)
        drawn_at = meta.get("llm_model", "an unrecorded model")
        print(
            f"fitting {len(candidates)} candidates read from {settings.proposals}, "
            f"a {meta.get('target', '?')} round from {name} drawn on {drawn_at}"
        )
    else:
        candidates = [Model(STARTS[settings.target, name])]
        origin = {0: name}
        try:
            if searching:
                settings.llm_model = settings.llm_model or served(settings.url)
                print(
                    f"asking {settings.agents} agents on {settings.llm_model} for "
                    f"a {settings.target} block, from {name}"
                )
                _kept, proposals = await propose(settings, candidates[0], brief, pool)
                drawn.append((name, proposals))
                for proposal in proposals:
                    if proposal.model is None:
                        continue
                    # Walked per proposal rather than over `kept` so the agent
                    # that wrote each candidate is kept alongside it; the order
                    # is the same either way.
                    placed[name, proposal.agent] = len(candidates)
                    origin[len(candidates)] = name
                    candidates.append(proposal.model)
        finally:
            # Before the fits rather than after them: the workers hold the table
            # and a few hundred megabytes each, and the fits want every core.
            if pool is not None:
                await pool.aclose()
        if failed := failures(drawn, settings.out):
            print(f"failed agents: {failed}")
        if kept_talk := transcripts(drawn, placed, settings.out):
            print(f"transcripts: {kept_talk}")

    if settings.propose_only:
        # A round of nothing but the model it started from is a failed round, and
        # writing it as a directory the fit half will happily consume is how a
        # server that was down for ten seconds becomes a report that says the
        # incumbent won. Refuse it here, where the reason is still on screen.
        if len(candidates) == 1:
            raise SystemExit(f"no proposals parsed: nothing to write. See {settings.out}/failures.jsonl")
        written = save(candidates, origin, settings, settings.out)
        print(f"proposals: {written}")
        print(f"fit them with --proposals {written} (no server needed)")
        return 0

    budget = Budget(
        table=str(inputs.table),
        eir_priors=str(inputs.eir_priors),
        mortality=inputs.mortality,
        splits=splits,
        holdout_seed=settings.holdout_seed,
        n_studies=settings.n_studies,
        warmup=settings.warmup,
        samples=settings.samples,
        chains=settings.chains,
        seed=settings.seed,
        max_parameters=settings.max_parameters,
        inc_max=settings.inc_max,
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
    fitted(settings, splits, budget)
    print(f"\n{table(evaluations, origin, starts, splits)}\n\n* the models the search started from")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
