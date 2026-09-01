"""The same round, drawn by a coding agent instead of by a chat completion.

    uv run python -m examples.malaria.agentic --out runs_agentic/01-claude/curves-griffin
    uv run python -m examples.malaria --proposals runs_agentic/01-claude/curves-griffin/proposals \\
        --out runs_agentic/01-claude/curves-griffin

Two halves, and this module is only the first. It draws N proposals against one
parent and writes them where `__main__.load` reads them, so the fitting, the
gates, the splits, the NUTS budget and the report are the *same code* the i.i.d.
arm goes through. Nothing about the evaluation is re-implemented here, and that
is the only reason the two arms are comparable at all.

What differs from `search.py` is the proposer, and only the proposer. There, N
samples come back from one request against a chat model. Here, N coding agents
each run in their own sandbox with a shell, a filesystem and the training rows,
and each writes a `proposal.py`. The draw is the same draw; the harness around it
is not.

## Isolation

An agent that can read this repository is not proposing a model, it is
remembering one -- `runs*/` holds every proposal a previous round made and
`wip_paper.pdf` holds the scores. So a sandbox is a fresh directory outside the
project, holding the parent model, the brief and the training rows and nothing
else. It is deliberately not a git repository: a worktree would carry the history
in with it.

The path itself is data. Claude Code's default system prompt states the working
directory, the environment and the git status, so a sandbox under
`.../projects/discovery/` names the project in the prompt before the agent has
read anything. Sandboxes therefore live under a neutral root and
`--exclude-dynamic-system-prompt-sections` moves what is left out of the cached
prefix.

Four more leaks are closed on the command line:

* `--bare` skips hooks, plugins, auto-memory and CLAUDE.md discovery, so this
  machine's notes about this project cannot reach the agent. It requires
  `ANTHROPIC_API_KEY` (OAuth and keychain are never read under `--bare`), which
  is why `--auth session` exists: it falls back to `--safe-mode`, which disables
  the same customisations but leaves authentication alone. `--bare` is the
  stronger isolation and the one to report.
* `--setting-sources ""` and a throwaway `CLAUDE_CONFIG_DIR` per agent, so no
  user, project or local settings file is read and no two agents share state.
* `--tools` names the toolset positively rather than denying one. It keeps
  `WebSearch` and `WebFetch` out -- the agent would otherwise find Griffin 2014,
  or this repository -- and, less obviously, it keeps `Task` out: an agent that
  can spawn subagents turns one draw into k draws and quietly breaks the outer
  count this arm is compared on.
* `--no-session-persistence`, so a round leaves nothing for the next one to
  resume.

None of this touches the leak that cannot be closed: the model was pretrained on
the malaria literature. At `--start griffin` that is moot, since the incumbent is
handed over anyway. Anywhere else it is a caveat for the write-up, not a flag.

## What the agent is given, and what it is not

The brief is `initial.BRIEF`, formatted with the same numbers `__main__` formats
it with, so both arms are told the same thing about the fold, the fit budget, the
scoring and the gates. The data is `analysis.build()` -- the same training rows
the `--analysis` bench serves, with the scored fold already removed -- written
out as a CSV and an npz with a loader that binds the same names, alongside a
venv holding numpy and pandas -- `analysis.py`'s "numpy and pandas only" written
out as an interpreter.

It is *not* given a gate checker, and it cannot fit. Neither is an oversight:

* a gate checker would make the arms differ in yield rather than in modelling.
  13 of 32 i.i.d. proposals survived to a fit, and an agent handed
  `gates.check` would survive at nearly 100% for reasons that have nothing to do
  with the curve family it wrote.
* an agent that can run NUTS and read `elpd_holdout` is doing best-of-N against
  an evaluator, which is a different algorithm from an i.i.d. draw, and it would
  be selecting on the held-out fold. The sandbox carries no jax, no numpyro and
  no msinf, and the fold's rows are not in it.

## Validation

The agent writes source, so it can edit outside the markers -- an i.i.d. proposer
can only claim to have. Nothing it writes is fitted as-is: `collect` takes the
block bodies out of `proposal.py` and splices them onto the parent, so the
likelihood, the observational model and the scoring scaffolding come from the
parent by construction. Whether the agent tampered with `model.py` is recorded
rather than punished; it is a fact about the arm worth reporting.

## Determinism

Neither `claude` nor `codex` exposes temperature, top-p or a seed, and batched
serving would not honour a seed if they did. So the decode is not a control here,
it is a measurement: run the round, then compare the proposals to each other.
`--agents 8` twice over the same parent is the diversity experiment, and
`usage.jsonl` is what says whether the two arms cost the same to draw.
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from discovery.model import Model, parse_blocks

from examples.malaria import analysis as setup
from examples.malaria import data
from examples.malaria.__main__ import FIT_MINUTES
from examples.malaria.fit import Budget
from examples.malaria.initial import BRIEF, START_NAMES, STARTS, TARGETS

#: a neutral root for sandboxes. Deliberately short and anonymous: the agent is
#: told its working directory, and `/home/raj/.../projects/discovery/sandbox` is
#: a sentence about the experiment.
SANDBOX_ROOT = Path(os.environ.get("AGENTIC_SANDBOX", "/tmp/mx"))

#: the tools an agent may use. An allowlist rather than a denylist, because what
#: this arm was allowed to do is a claim the write-up has to make precisely --
#: and because the absence of `Task` here is what keeps one draw one draw.
TOOLS = "Bash,Read,Write,Edit,Glob,Grep"

#: the whole of the agent's python, from PyPI. `analysis.py`'s "numpy and pandas
#: only" written out as a venv.
BENCH_PACKAGES = ("numpy", "pandas")

#: and `dmeq`, copied in rather than installed, because installing it would drag
#: jax in with it -- it declares jax as a dependency but imports it only when a
#: backend is selected, which is the same trick `analysis.py` relies on to use
#: dmeq inside a pandas-only worker.
#:
#: Without it the sandbox cannot execute `model.py` at all: the file opens
#: `from dmeq import backend, default_parameters`, so an agent that wants to see
#: what the incumbent does has to reimplement it -- and then reasons about a
#: replica rather than about the model that will actually be fitted. Observed
#: doing exactly that, at a cost of twelve minutes and a silent risk of
#: optimising the wrong thing.
#:
#: `numpyro` stays out, and with it any way to fit: it needs jax. So `parameters()`
#: does not execute here and `mechanism` does, given a plain dict of values --
#: which is what the contract says it takes anyway.
BENCH_MODULES = ("dmeq",)

#: where that interpreter lands in every sandbox, and what the agent is told to use
VENV = ".venv"

#: what the agent is asked to be, appended to the coding harness's own system
#: prompt. The first paragraph is `proposer.SYSTEM_PROMPT`, changed only where a
#: file has replaced a message.
#:
#: The budget is stated because an agent cannot see the clock. Told only what it
#: may do, it explores until the harness kills it: observed spending forty
#: minutes building a likelihood and an optimiser of its own and being cut off
#: with no proposal written, having never known there was a deadline. An
#: unannounced timeout does not measure how good a proposer is, it measures how
#: cautious one happened to be.
ROLE = """\
You are a scientist searching for a better model. Models are programs.

The program is `model.py` in your working directory. Regions you may rewrite are
delimited by `# @discovery:begin <name>` and `# @discovery:end` comments.
Everything outside those markers is fixed: it is the harness that runs and scores
the model, and you must not alter it. Do not edit `model.py`.

Write your answer to `proposal.py`: only the rewritten regions, each still
wrapped in its own begin/end markers and keeping its original name, and nothing
else in the file. Propose a genuine change: a different functional form, a
different distribution, a new term or structure, not a cosmetic edit.

`BRIEF.md` says what the model will be fitted to and scored on. `DATA.md`
describes the training rows, which are in `data/` -- the rows this round is
scored on are not there, and no amount of looking will reveal them. Run whatever
you like against that data with `./.venv/bin/python`, which has numpy, pandas and
`dmeq` in it. Use that interpreter and not a bare `python`.

`dmeq` is there so that the mechanism in `model.py` runs here as it is written,
on dmeq's numpy backend. Do not reimplement it to look at it: a replica you wrote
is not the model that gets scored, and tuning against one is how you end up
confident about a structure that behaves differently when it is actually fitted.

`numpyro` is not installed -- it needs jax, and jax is how you would fit -- so
`model.py` will not import as it stands, and `parameters()` will not run. Stub it
and the rest of the file works:

    import sys, types, numpy as np
    sys.modules["numpyro"] = types.ModuleType("numpyro")
    sys.modules["numpyro"].sample = lambda name, d: None
    sys.modules["numpyro.distributions"] = types.ModuleType("numpyro.distributions")

    ns = {"__name__": "candidate"}
    exec(compile(open("model.py").read(), "model.py", "exec"), ns)

    from dmeq import use_backend, default_parameters
    params = dict(default_parameters()) | {...}   # the priors' medians are in the file
    with use_backend("numpy"):
        prev_m, inc = ns["mechanism"](params, np.log(eir), ft, ages, prop_row, mu_row)

`params` is a plain dict, which is what the contract says `mechanism` takes, so
nothing is lost by not being able to sample it. The same recipe runs whatever you
write into `proposal.py`, spliced back into the file.

`msinf` is not here: the likelihood, the split and the score are not yours to
run, and this sandbox cannot fit anything. There is no network.

## Your budget

You have @@MINUTES@@ minutes of wall clock, and nothing warns you as it runs
out: when it is gone the session is killed wherever it has got to, and a session
with no `proposal.py` in it has produced nothing at all.

So write a `proposal.py` you would be content to be scored on early, well inside
the first half of that, and improve it afterwards. Overwrite it as often as you
like -- only the last version is taken. Investigate as much as the time allows;
just do not let the deadline arrive with the answer still unwritten.\
"""

#: the user turn, which is `ModelProposer.prompt` with the model and the feedback
#: moved into files. Same three things in the same order: the model, what it will
#: be scored on, the instruction.
TASK = """\
Here is the current model: `model.py`.

Rewritable regions: {names}.

Results for this model: see `BRIEF.md`. The training rows are described in
`DATA.md` and loaded by `data/load.py`.

Propose an improved version of the rewritable regions, and write them to
`proposal.py` as described. When `proposal.py` is written, you are done.\
"""

#: how the training rows are presented, since the agent loads them itself rather
#: than being handed a namespace. Everything after the first paragraph is
#: `analysis.DESCRIPTION` verbatim, so both arms read the same description of the
#: same columns.
DATA_DOC = """\
# The training rows

`data/load.py` binds the namespace below. From this directory:

    ./.venv/bin/python -c "import sys; sys.path.insert(0, 'data'); from load import *; print(df.head())"

or `from load import df, ages, prop, grid, re, eps, np, pd` in a script you put
anywhere in the sandbox. Everything is loaded from `data/train.csv` and
`data/arrays.npz`; nothing you do to the frame is written back.

{description}\
"""

#: the loader that lands in the sandbox. It binds exactly the names
#: `analysis.namespace()` binds, so a snippet written for one arm's bench runs in
#: the other's sandbox unchanged.
LOADER = '''\
"""The training rows and the fixed model inputs, under the names the brief uses.

The scored fold is already gone: this file loads what was written into the
sandbox, and what was written into the sandbox is what was left after the split.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

df = pd.read_csv(HERE / "train.csv")

_arrays = np.load(HERE / "arrays.npz")

#: the model's age grid: 165 class lower edges in years
ages = _arrays["ages"]
#: (n_site, n_age) population fraction per age class, indexed by `site_i`
prop = _arrays["prop"]
#: (n_site, n_age) per-day rate of leaving an age class: ageing plus mortality
re = _arrays["re"]
#: (n_site, n_age) per-day EIR each age class experiences at the prior central EIR
eps = _arrays["eps"]

#: the same geometry `dmeq.age_grid` builds, as an attribute bag
grid = SimpleNamespace(
    years=_arrays["grid_years"],
    days=_arrays["grid_days"],
    widths=_arrays["grid_widths"],
    midpoints=_arrays["grid_midpoints"],
    ageing=_arrays["grid_ageing"],
    age20=int(_arrays["grid_age20"]),
)

__all__ = ["df", "ages", "prop", "grid", "re", "eps", "np", "pd"]
'''


def sandboxes(root: Path, agents: int) -> list[Path]:
    """Where each agent works, in agent order."""
    return [root / f"a{agent:02d}" for agent in range(agents)]


def venv(path: Path) -> None:
    """Give one sandbox its own python, holding numpy and pandas and nothing else.

    Per agent rather than one shared between them: an agent has a shell, so a
    shared environment is a channel between agents -- one that installs a package
    or drops a file in `site-packages` changes what the others see. `uv` hardlinks
    from its cache, so eight of these cost about as much as one.
    """
    target = path / VENV
    if (target / "bin" / "python").exists():
        return
    subprocess.run(["uv", "venv", "--quiet", str(target)], check=True)
    subprocess.run(
        ["uv", "pip", "install", "--quiet", "--python", str(target / "bin" / "python"), *BENCH_PACKAGES],
        check=True,
    )
    site = next(iter(target.glob("lib/python*/site-packages")))
    for name in BENCH_MODULES:
        module = __import__(name)
        shutil.copytree(Path(module.__file__).parent, site / name, dirs_exist_ok=True)


def arrays(settings: argparse.Namespace) -> dict:
    """The fixed model inputs, from the bench the i.i.d. arm reads.

    `analysis.build` is the function that owns the fold -- it draws the same
    split the fit will draw, through `msinf.splits`, and returns what is left.
    Calling it rather than re-deriving the training rows here is the whole
    guarantee that the sandbox cannot contain a scored row.
    """
    inputs = data.inputs(settings.table, settings.eir_priors, settings.mortality)
    os.environ["MALARIA_TABLE"] = str(inputs.table)
    os.environ["MALARIA_EIR_PRIORS"] = str(inputs.eir_priors)
    os.environ["MALARIA_MORTALITY"] = inputs.mortality or ""
    os.environ["MALARIA_SPLITS"] = ",".join(settings.splits)
    os.environ["MALARIA_HOLDOUT_SEED"] = str(settings.holdout_seed)
    os.environ["MALARIA_N_STUDIES"] = str(settings.n_studies)

    train, ages, prop, grid, re, eps = setup.build()
    return {
        "train": train,
        "npz": {
            "ages": ages,
            "prop": prop,
            "re": re,
            "eps": eps,
            "grid_years": grid.years,
            "grid_days": grid.days,
            "grid_widths": grid.widths,
            "grid_midpoints": grid.midpoints,
            "grid_ageing": grid.ageing,
            "grid_age20": grid.age20,
        },
        "summary": data.read(inputs.table),
    }


def brief(settings: argparse.Namespace, summary: str) -> str:
    """`initial.BRIEF`, formatted with the numbers `__main__` formats it with.

    Duplicated argument by argument rather than imported as a formatted string,
    because the two arms have to be told the same thing and the only way to see
    that they are is for the call to look like the other call. If `__main__`
    grows a field here, this fails loudly at `KeyError` rather than quietly
    telling one arm less than the other.
    """
    return BRIEF.format(
        table=summary,
        study=settings.n_studies,
        chains=settings.chains,
        warmup=settings.warmup,
        samples=settings.samples,
        minutes=FIT_MINUTES,
        inc_max=Budget.inc_max,
    )


def role(settings: argparse.Namespace) -> str:
    """`ROLE`, with the deadline the runner will actually enforce written into it.

    Taken from `--session-timeout` rather than restated, so the number the agent
    is told and the number that kills it cannot drift apart.

    A sentinel rather than `str.format`, for the reason `initial.SKELETON` gives:
    what is spliced around it is Python, and Python is full of braces.
    """
    return ROLE.replace("@@MINUTES@@", f"{settings.session_timeout / 60.0:.0f}")


def build(settings: argparse.Namespace) -> list[Path]:
    """Write one sandbox per agent, and return them in agent order.

    Every sandbox is byte-identical apart from its path: the agents are meant to
    be N draws from one prompt, so anything that differed between them would be
    an intervention the i.i.d. arm never got.
    """
    import numpy as np

    parent = Model(STARTS[(settings.target, settings.start)])
    names = ", ".join(block.name for block in parent.blocks) or "(none)"
    built = arrays(settings)
    text = brief(settings, built["summary"])

    root = settings.sandbox
    if root.exists() and settings.clean:
        shutil.rmtree(root)
    made: list[Path] = []
    for path in sandboxes(root, settings.agents):
        (path / "data").mkdir(parents=True, exist_ok=True)
        (path / "model.py").write_text(parent.source)
        (path / "BRIEF.md").write_text(f"# What this model will be scored on\n\n{text}")
        (path / "DATA.md").write_text(DATA_DOC.format(description=setup.DESCRIPTION))
        (path / "PROMPT.md").write_text(TASK.format(names=names) + "\n")
        (path / "ROLE.md").write_text(role(settings) + "\n")
        built["train"].to_csv(path / "data" / "train.csv", index=False)
        np.savez(path / "data" / "arrays.npz", **built["npz"])
        (path / "data" / "load.py").write_text(LOADER)
        venv(path)
        # A stale proposal from an earlier attempt would be collected as this
        # round's answer, which is the one failure mode that would not announce
        # itself.
        (path / "proposal.py").unlink(missing_ok=True)
        made.append(path)
    return made


def command(settings: argparse.Namespace, path: Path) -> list[str]:
    """The argv for one agent, written beside its log so the paper can quote it.

    `--bare` and `--safe-mode` are the two isolation levels, and they differ in
    where authentication comes from rather than in what they disable: `--bare`
    reads `ANTHROPIC_API_KEY` only, `--safe-mode` leaves the normal login alone.
    Prefer `--bare`.
    """
    argv = [
        settings.cli,
        "-p",
        "--bare" if settings.auth == "api-key" else "--safe-mode",
        "--model", settings.model,
        "--setting-sources", "",
        "--strict-mcp-config",
        "--tools", TOOLS,
        "--append-system-prompt", (path / "ROLE.md").read_text(),
        "--exclude-dynamic-system-prompt-sections",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
    ]
    if settings.max_budget_usd:
        argv += ["--max-budget-usd", str(settings.max_budget_usd)]
    return argv


def environment(settings: argparse.Namespace, path: Path) -> dict[str, str]:
    """The agent's environment: this machine's, minus anything about this project.

    `CLAUDE_CONFIG_DIR` is redirected under `--bare` so nothing in the real one is
    read and no two agents share a config. Under `--safe-mode` it is left where it
    is, because that is where the login lives -- the customisations are disabled
    by the flag instead.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE_", "MALARIA_", "DISCOVERY_"))}
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    if settings.auth == "api-key":
        config = path.parent / f"cfg-{path.name}"
        config.mkdir(parents=True, exist_ok=True)
        env["CLAUDE_CONFIG_DIR"] = str(config)
    return env


async def session(settings: argparse.Namespace, agent: int, path: Path, out: Path) -> dict:
    """Run one agent to completion and write its raw stream to `out`.

    Nothing is raised for an agent that fails: a session that crashed, timed out
    or never wrote a proposal costs that agent and nothing else, exactly as a
    malformed reply costs one sample of an i.i.d. batch.
    """
    argv = command(settings, path)
    # Beside the log and not in the sandbox: an agent reads what is in front of
    # it, and one that ran `cat command.txt` learned its own argv, its isolation
    # flags and the fact that it was in an experiment at all.
    (out / f"a{agent:02d}.command.txt").write_text("\n".join(argv) + "\n")
    prompt = (path / "PROMPT.md").read_text()
    started = time.perf_counter()
    log = out / f"a{agent:02d}.jsonl"
    with open(log, "w") as handle:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=path,
            env=environment(settings, path),
            stdin=asyncio.subprocess.PIPE,
            stdout=handle,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode()), timeout=settings.session_timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "agent": agent,
                "returncode": None,
                "error": f"the session did not finish within {settings.session_timeout:.0f}s",
                "seconds": time.perf_counter() - started,
                "log": str(log),
            }
    return {
        "agent": agent,
        "returncode": process.returncode,
        "error": None if process.returncode == 0 else (stderr or b"").decode()[-3000:],
        "seconds": time.perf_counter() - started,
        "log": str(log),
    }


async def run(settings: argparse.Namespace, paths: list[Path], out: Path) -> list[dict]:
    """Run every agent, at most `--concurrency` at once."""
    out.mkdir(parents=True, exist_ok=True)
    limit = asyncio.Semaphore(settings.concurrency)

    async def one(agent: int, path: Path) -> dict:
        async with limit:
            print(f"  agent {agent}: starting in {path}")
            result = await session(settings, agent, path, out)
            note = result["error"] or "ok"
            print(f"  agent {agent}: {note.splitlines()[0] if note else 'ok'} ({result['seconds']:.0f}s)")
            return result

    return list(await asyncio.gather(*(one(agent, path) for agent, path in enumerate(paths))))


def rendered(block: dict) -> str:
    """One content block of a stream-json message, as text.

    Tool calls and their results are kept rather than summarised away: for an
    agent they are most of what happened, and `transcripts.jsonl` exists so that
    a candidate can be traced back to what its author actually did.
    """
    kind = block.get("type")
    if kind == "text":
        return block.get("text", "")
    if kind == "thinking":
        return f"[thinking]\n{block.get('thinking', '')}"
    if kind == "tool_use":
        return f"[tool_use {block.get('name')}]\n{json.dumps(block.get('input', {}), default=str)}"
    if kind == "tool_result":
        content = block.get("content")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        return f"[tool_result]\n{content}"
    return json.dumps(block, default=str)


def conversation(log: Path) -> tuple[list[dict], dict]:
    """One session's turns in `transcripts.jsonl`'s shape, and what it cost.

    The stream is the record; this is a reading of it that the existing tooling
    can join on. The raw file stays beside it, because a rendering is lossy and
    the cost of keeping both is a few megabytes.
    """
    turns: list[dict] = []
    usage: dict = {}
    resolved: str | None = None
    if not log.exists():
        return turns, usage
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind in {"assistant", "user"}:
            message = event.get("message", {})
            resolved = message.get("model") or resolved
            content = message.get("content")
            if isinstance(content, list):
                content = "\n\n".join(rendered(block) for block in content if isinstance(block, dict))
            turns.append({"role": message.get("role", kind), "content": content or ""})
        elif kind == "result":
            counts = event.get("usage") or {}
            # Per model, because a session is not one model: the harness runs a
            # small one alongside for its own background work, and a claim about
            # what a draw cost has to say so rather than quietly attribute those
            # tokens to the proposer. `webSearchRequests` is in here too, and it
            # is the audit trail for the tool that is not in `TOOLS`: anything
            # other than zero means an agent reached the internet.
            per_model = event.get("modelUsage") or {}
            usage = {
                "input_tokens": counts.get("input_tokens"),
                "output_tokens": counts.get("output_tokens"),
                "cache_read_input_tokens": counts.get("cache_read_input_tokens"),
                "cache_creation_input_tokens": counts.get("cache_creation_input_tokens"),
                "total_cost_usd": event.get("total_cost_usd"),
                "duration_ms": event.get("duration_ms"),
                "duration_api_ms": event.get("duration_api_ms"),
                "num_turns": event.get("num_turns"),
                "is_error": event.get("is_error"),
                "subtype": event.get("subtype"),
                "model_usage": per_model,
                "web_searches": sum(
                    (m or {}).get("webSearchRequests", 0) for m in per_model.values()
                ),
            }
    if resolved:
        usage["model"] = resolved
    return turns, usage


class ProposalError(ValueError):
    """A session that produced nothing that could be spliced onto the parent."""


def validate(parent: Model, path: Path) -> tuple[Model, dict]:
    """Splice one agent's `proposal.py` onto the parent, or say why it cannot be.

    The parent is the source of everything outside the markers, always. What the
    agent wrote is read for block bodies and for nothing else, so an agent that
    rewrote the likelihood, deleted a gate or renamed a function the harness calls
    has changed nothing -- the edit simply does not survive the splice. That an
    agent tried is recorded, because it is a fact about this arm and one the
    i.i.d. arm could not have exhibited.
    """
    note = {
        "tampered": (path / "model.py").read_text() != parent.source,
        "proposal": (path / "proposal.py").exists(),
    }
    if not note["proposal"]:
        raise ProposalError("the agent wrote no proposal.py")
    text = (path / "proposal.py").read_text()
    bodies = parse_blocks(text)
    if not bodies:
        raise ProposalError("proposal.py contains no @discovery blocks")
    known = {block.name for block in parent.blocks}
    if unknown := set(bodies) - known:
        raise ProposalError(f"proposal.py names unknown block(s) {sorted(unknown)}")
    candidate = parent.substitute(bodies)
    if candidate.source == parent.source:
        raise ProposalError("the proposal is identical to the model it started from")
    try:
        compile(candidate.source, "<candidate>", "exec")
    except SyntaxError as error:
        raise ProposalError(f"the spliced candidate does not parse: {error}") from error
    note["blocks"] = sorted(bodies)
    return candidate, note


def collect(settings: argparse.Namespace, paths: list[Path], out: Path, sessions: list[dict]) -> int:
    """Turn N finished sandboxes into a round `__main__ --proposals` can fit.

    Same layout `__main__.save` writes: `0000-<parent>.py` is the model every
    agent was given, unmodified, and every kept proposal follows it in agent
    order. Beside it go the three records that make the round auditable --
    the conversations, the failures, and what each draw cost.
    """
    parent = Model(STARTS[(settings.target, settings.start)])
    directory = out / "proposals"
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.py"):
        old.unlink()
    (directory / f"0000-{settings.start}.py").write_text(parent.source)

    by_agent = {record["agent"]: record for record in sessions}
    kept: list[dict] = []
    failed: list[dict] = []
    records: list[dict] = []
    usages: list[dict] = []
    index = 0
    resolved: str | None = None

    for agent, path in enumerate(paths):
        session_record = by_agent.get(agent, {})
        log = Path(session_record.get("log", out / "sessions" / f"a{agent:02d}.jsonl"))
        turns, usage = conversation(log)
        resolved = usage.get("model") or resolved
        error: str | None = None
        note: dict = {}
        try:
            candidate, note = validate(parent, path)
        except ProposalError as failure:
            error = f"{type(failure).__name__}: {failure}"
        except Exception as failure:  # a sandbox that is not there, an unreadable file
            error = f"{type(failure).__name__}: {failure}"
        else:
            index += 1
            (directory / f"{index:04d}-{settings.start}.py").write_text(candidate.source)
            kept.append({"agent": agent, "candidate": index})

        if error is not None:
            failed.append(
                {
                    "start": settings.start,
                    "agent": agent,
                    "error": error,
                    "reply": (path / "proposal.py").read_text() if (path / "proposal.py").exists() else None,
                    "session": session_record.get("error"),
                    "exchange": turns,
                }
            )
        records.append(
            {
                "start": settings.start,
                "agent": agent,
                "ok": error is None,
                "candidate": index if error is None else None,
                "error": error,
                "tampered": note.get("tampered"),
                "blocks": note.get("blocks"),
                "transcript": turns,
            }
        )
        usages.append(
            {
                "agent": agent,
                "ok": error is None,
                "seconds": session_record.get("seconds"),
                "returncode": session_record.get("returncode"),
                **usage,
            }
        )

    write_lines(out / "transcripts.jsonl", records)
    write_lines(out / "usage.jsonl", usages)
    if failed:
        write_lines(out / "failures.jsonl", failed)

    (directory / "round.json").write_text(
        json.dumps(
            {
                "target": settings.target,
                "start": settings.start,
                # what `sweep.py` prints as the model behind the round: the id the
                # server actually resolved, not the alias asked for, because an
                # alias moves and a round has to say what drew it.
                "llm_model": resolved or settings.model,
                "requested_model": settings.model,
                "harness": "claude-code",
                # `message.model` comes back as the alias itself -- a first-party
                # session reports `claude-opus-5`, not a dated id -- so what
                # actually pins this round is the alias, the CLI version and the
                # day it was drawn, together.
                "drawn_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "cli_version": version(settings.cli),
                "tools": TOOLS,
                "auth": settings.auth,
                "agents": settings.agents,
                "analysis": True,
                "tool_budget": None,
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "max_tokens": None,
                "thinking_token_budget": None,
                "candidates": index + 1,
                "tampered": sum(1 for r in records if r.get("tampered")),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  {len(kept)}/{len(paths)} proposals validated -> {directory}")
    if tampered := sum(1 for r in records if r.get("tampered")):
        print(f"  {tampered} agent(s) edited model.py; the edits did not survive the splice")
    return index


def write_lines(path: Path, records: list[dict]) -> Path:
    """One json object per line, the shape the rest of the example reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for record in records:
            json.dump(record, handle, default=str)
            handle.write("\n")
    return path


def version(cli: str) -> str | None:
    """The coding harness's own version, which pins the round as much as the model does."""
    try:
        return subprocess.run(
            [cli, "--version"], capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="examples.malaria.agentic", description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs_agentic/01-claude/curves-griffin"),
                        help="two levels under the sweep root, so sweep.py finds it")
    parser.add_argument("--agents", type=int, default=8, help="proposals drawn, one agent each")
    parser.add_argument("--target", default="curves", choices=TARGETS)
    parser.add_argument("--start", default="griffin", choices=START_NAMES)

    harness = parser.add_argument_group("coding harness")
    harness.add_argument("--cli", default="claude")
    harness.add_argument("--model", default="claude-opus-5",
                         help="pinned alias or full id; the id it resolves to is what round.json records")
    harness.add_argument("--auth", default="api-key" if os.environ.get("ANTHROPIC_API_KEY") else "session",
                         choices=("api-key", "session"),
                         help="api-key uses --bare (stronger isolation); session uses --safe-mode")
    harness.add_argument("--sandbox", type=Path, default=None,
                         help=f"neutral root for the agents' directories (default under {SANDBOX_ROOT})")
    harness.add_argument("--concurrency", type=int, default=4, help="agents running at once")
    harness.add_argument("--session-timeout", type=float, default=3600.0, help="seconds one agent may take")
    harness.add_argument("--max-budget-usd", type=float, default=5.0, help="0 to leave it unbounded")
    harness.add_argument("--keep", action="store_true", help="do not clear the sandbox root first")

    inputs = parser.add_argument_group("what the brief describes")
    inputs.add_argument("--table", default=str(data.DEFAULT_TABLE))
    inputs.add_argument("--eir-priors", default=str(data.DEFAULT_EIR_PRIORS))
    inputs.add_argument("--mortality", default=data.DEFAULT_MORTALITY)
    inputs.add_argument("--splits", default=",".join(data.SPLITS))
    inputs.add_argument("--holdout-seed", type=int, default=42)
    inputs.add_argument("--n-studies", type=int, default=3)
    inputs.add_argument("--warmup", type=int, default=400)
    inputs.add_argument("--samples", type=int, default=400)
    inputs.add_argument("--chains", type=int, default=4)

    phases = parser.add_argument_group("phases")
    phases.add_argument("--build-only", action="store_true", help="write the sandboxes and stop")
    phases.add_argument("--collect-only", action="store_true",
                        help="validate sandboxes an earlier run left, without running anything")

    settings = parser.parse_args(argv)
    settings.splits = tuple(s.strip() for s in settings.splits.split(",") if s.strip())
    if unknown := set(settings.splits) - set(data.SPLITS):
        raise SystemExit(f"unknown split(s) {sorted(unknown)}; expected some of {list(data.SPLITS)}")
    if settings.sandbox is None:
        settings.sandbox = SANDBOX_ROOT / f"{settings.target[:1]}{settings.start[:1]}-{os.getpid():05d}"
    settings.clean = not settings.keep
    return settings


async def main(argv: list[str] | None = None) -> int:
    settings = options(argv)
    if settings.auth == "api-key" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("--auth api-key needs ANTHROPIC_API_KEY; use --auth session to use the normal login")
    if shutil.which(settings.cli) is None:
        raise SystemExit(f"no {settings.cli!r} on PATH")

    out = settings.out
    print(f"target  {settings.target}")
    print(f"start   {settings.start}")
    print(f"model   {settings.model} via {settings.cli} ({version(settings.cli) or 'unknown version'})")
    print(f"split   {data.label(settings.splits[0], settings.holdout_seed, settings.n_studies)}")

    if settings.collect_only:
        paths = sandboxes(settings.sandbox, settings.agents)
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"--collect-only found no sandbox at {missing[0]}")
    else:
        paths = build(settings)
        print(f"built   {len(paths)} sandboxes under {settings.sandbox}")
        if settings.build_only:
            return 0

    sessions: list[dict] = []
    if not settings.collect_only:
        sessions = await run(settings, paths, out / "sessions")
        write_lines(out / "sessions.jsonl", sessions)

    collect(settings, paths, out, sessions)
    print()
    print(f"fit it with: uv run python -m examples.malaria --proposals {out / 'proposals'} --out {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
