"""The same round, drawn by a coding agent instead of by a chat completion.

    uv run python -m examples.malaria.opencode --out runs/beat/r02 --history runs/beat
    uv run python -m examples.malaria.codex --out runs/beat/r03 --history runs/beat
    uv run python -m examples.malaria --proposals runs/beat/r02/proposals --out runs/beat/r02

`examples/README.md`, "Searching for a better model", has the whole loop,
cluster fits included.

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

`--cli opencode` drives a local OpenAI-compatible server through OpenCode V2,
and closes the same leaks differently, because its flags do not reach them:

* each agent gets a throwaway `HOME`, `XDG_CONFIG_HOME` and `XDG_DATA_HOME`
  beside its sandbox, holding the only `opencode.json` it reads and the session
  database. The `HOME` is not cosmetic: OpenCode loads skills from
  `~/.claude/skills` and `~/.agents/skills` whatever the config says.
* `OPENCODE_DISABLE_PROJECT_CONFIG=1`, so no `AGENTS.md` above the sandbox is
  read. `ROLE` goes in the per-agent global `AGENTS.md` instead, which is added
  after OpenCode's own system prompt -- the analogue of `--append-system-prompt`.
  An agent's `system` field would *replace* that prompt, which the other arms
  never do.
* `--standalone`, so every agent has a private server rather than sharing the
  user's background service, which owns sessions, configuration and tool
  execution.
* permission rules deny web fetch, web search, subagents, skills, questions and
  paths outside the sandbox. `--auto` approves anything left at `ask`, so a rule
  that only asked would be no rule at all. The shell is the exception: OpenCode
  infers a command's directories on a best-effort basis, and `cat /etc/hostname`
  was observed to pass. That is the same exposure Claude Code's `Bash` has here,
  and `outside_workspace` is the audit for both.

None of this touches the leak that cannot be closed: the model was pretrained on
the malaria literature. At `--start griffin` that is moot, since the incumbent is
handed over anyway. Anywhere else it is a caveat for the write-up, not a flag.

## What the agent is given, and what it is not

The brief is `initial.BRIEF`, formatted with the same numbers `__main__` formats
it with, so both arms are told the same thing about the fold, the fit budget, the
scoring and the gates. The data is `analysis.build()` -- the same training rows
the `--analysis` bench serves, with the scored fold already removed -- written
out as a CSV and an npz with a loader that binds the same names, alongside a
venv holding numpy, pandas, jax and numpyro.

This arm was first built to be *compared* with the i.i.d. one, and so was given
no gate checker, no way to fit and no results. The search is now for a better
model rather than a fair comparison, so it is given all three:

* `check.py`, which splices `proposal.py` onto `model.py` and runs `gates.check`
  exactly as the runner does, so no fit is spent on a candidate that a gate
  would have refused;
* jax and numpyro, so an agent can fit whatever it likes to the training rows;
* `feedback/`, written by `feedback.py` from every round already fitted
  (`--history`): the leaderboard, each candidate's fitted training rows, and why
  the failures failed.

One thing it is never given, whatever else changes: the held-out fold. Not its
rows, not any candidate's score on them, not a posterior or a picture from a fit
that saw them. The ranking is decided on 35 held-out units, and an agent that
could tune against them round after round would make that ranking mean nothing.
msinf stays out too -- it would point at the repository and its full table --
so an agent's own fits are its own, and only msinf's fit of the spliced
candidate is scored.

## Validation

The agent writes source, so it can edit outside the markers -- an i.i.d. proposer
can only claim to have. Nothing it writes is fitted as-is: `collect` takes the
block bodies out of `proposal.py` and splices them onto the parent, so the
likelihood, the observational model and the scoring scaffolding come from the
parent by construction. Whether the agent tampered with `model.py` is recorded
rather than punished; it is a fact about the arm worth reporting.

## Determinism

Neither `claude`, `codex` nor `opencode run` exposes temperature, top-p or a
seed, and batched serving would not honour a seed if they did. So the decode is not a control here,
it is a measurement: run the round, then compare the proposals to each other.
`--agents 8` twice over the same parent is the diversity experiment, and
`usage.jsonl` is what says whether the two arms cost the same to draw.
"""

import argparse
import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from contextlib import closing
from importlib.metadata import version as installed
from pathlib import Path

import discovery.model
from discovery.model import Model, parse_blocks

from examples.malaria import analysis as setup
from examples.malaria import data, feedback
from examples.malaria.__main__ import FIT_MINUTES
from examples.malaria.fit import ROOT, Budget
from examples.malaria.initial import BRIEF, START_NAMES, STARTS, TARGETS

#: a neutral root for sandboxes. Deliberately short and anonymous: the agent is
#: told its working directory, and `/home/raj/.../projects/discovery/sandbox` is
#: a sentence about the experiment.
SANDBOX_ROOT = Path(os.environ.get("AGENTIC_SANDBOX", "/tmp/mx"))

#: the tools an agent may use. An allowlist rather than a denylist, because what
#: this arm was allowed to do is a claim the write-up has to make precisely --
#: and because the absence of `Task` here is what keeps one draw one draw.
TOOLS = "Bash,Read,Write,Edit,Glob,Grep"

#: what `round.json` calls each CLI, which is what `frontier.py` reads an arm by
HARNESSES = {"claude": "claude-code", "codex": "codex-cli", "opencode": "opencode"}

#: the provider ID the per-agent OpenCode config defines, so a model is
#: referenced as `local/<the id the server serves>`.
OPENCODE_PROVIDER = "local"

#: the repository's own OpenCode configuration: what the local model is, how much
#: it can hold and which reasoning variants it has. Each agent's config is built
#: from it rather than restating it, so the two cannot drift.
OPENCODE_CONFIG = ROOT / "opencode.json"

#: tokens one OpenCode reply may use when neither the flag nor `opencode.json` says
OUTPUT_LIMIT = 16384

#: how long one agent may run, by CLI. Hours for the local and subscription
#: models: the point now is to let them explore, and this is a guard against a
#: hung session rather than a budget. The agent is told the number.
SESSION_TIMEOUTS = {"claude": 3600.0, "codex": 14400.0, "opencode": 14400.0}

#: the model each subscription CLI draws on unless `--model` says otherwise.
#: OpenCode has none: it asks the local server what it serves.
DEFAULT_MODELS = {"claude": "claude-opus-5", "codex": "gpt-6-luna"}

#: OpenCode's version of `TOOLS`. Its tools cannot be named positively, so this
#: denies what `TOOLS` leaves out and keeps the rest. The base policy every agent
#: starts from allows everything and asks only for external paths and `.env`
#: reads; the last matching rule wins, so these override it. `question` is denied
#: because nobody is there to answer it.
OPENCODE_PERMISSIONS = [
    {"action": "external_directory", "resource": "*", "effect": "deny"},
    {"action": "webfetch", "resource": "*", "effect": "deny"},
    {"action": "websearch", "resource": "*", "effect": "deny"},
    {"action": "subagent", "resource": "*", "effect": "deny"},
    {"action": "skill", "resource": "*", "effect": "deny"},
    {"action": "question", "resource": "*", "effect": "deny"},
]

#: how many times an OpenCode session that failed before its first model step is
#: started again. A standalone server occasionally reports the configured model
#: unavailable at startup; nothing has run in the sandbox by then, so a retry is
#: the same draw rather than a second one.
OPENCODE_RETRIES = 2

#: the agent's python, from PyPI, at the versions this project runs. jax and
#: numpyro so `check.py` runs the gates the runner runs, on the same numerics,
#: and so an agent can fit what it likes to the training rows.
BENCH_PACKAGES = ("numpy", "pandas", "jax", "jaxlib", "numpyro")

#: and `dmeq`, copied in rather than installed, because installing it from its
#: git source is slow and its pinned revision is whatever this environment holds.
#: Without it the sandbox cannot execute `model.py` at all: the file opens
#: `from dmeq import backend, default_parameters`, so an agent that wants to see
#: what the incumbent does has to reimplement it -- and then reasons about a
#: replica rather than about the model that will actually be fitted. Observed
#: doing exactly that, at a cost of twelve minutes and a silent risk of
#: optimising the wrong thing.
#:
#: msinf is not here, deliberately: see the module docstring.
BENCH_MODULES = ("dmeq",)

#: single files copied into the venv beside dmeq, as `{path in site-packages:
#: source}`: the gates, and the block grammar `check.py` splices with -- the
#: same two files the runner and `collect` use, so a pass here is a pass there.
BENCH_FILES = {
    "gates.py": Path(__file__).resolve().parent / "gates.py",
    "discovery/__init__.py": None,
    "discovery/model.py": Path(discovery.model.__file__),
}

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
you like against that data with `./.venv/bin/python`, which has numpy, pandas,
jax, numpyro and `dmeq` in it. Use that interpreter and not a bare `python`.

If there is a `feedback/` directory, read `feedback/FEEDBACK.md` first. It is what
earlier rounds proposed, how each of them fitted these same training rows, and
why the ones that failed did -- start from what it says rather than from scratch.

`dmeq` is there so that the mechanism in `model.py` runs here as it is written.
Do not reimplement it to look at it: a replica you wrote is not the model that
gets scored, and tuning against one is how you end up confident about a
structure that behaves differently when it is actually fitted. To run it:

    import jax; jax.config.update("jax_enable_x64", True)   # the fit runs at f64
    import numpy as np
    from numpyro import handlers
    from dmeq import use_backend

    ns = {"__name__": "candidate"}
    exec(compile(open("model.py").read(), "model.py", "exec"), ns)
    params = handlers.seed(ns["parameters"], rng_seed=0)()   # one draw from the prior
    params = {k: np.asarray(v) for k, v in params.items()}
    with use_backend("numpy"):
        prev_m, inc = ns["mechanism"](params, np.log(eir), ft, ages, prop_row, mu_row)

The same recipe runs whatever you write into `proposal.py`, spliced back into
the file.

Before you finish, run `./.venv/bin/python check.py`. It splices `proposal.py`
onto `model.py` and runs the gates every candidate must pass before it is
fitted -- the same checks, in the same code, that the harness runs. A candidate
that fails one is never fitted, so fix it rather than submit it.

jax and numpyro are here so you can fit what you like to the training rows -- an
optimisation, a short chain -- if that helps you choose. Your fits are your own:
the score is the harness's fit of your spliced candidate, under an observational
model that is not in this sandbox (study random effects, a beta-binomial for
prevalence, a Poisson with a site-age effect for incidence), scored on studies
you cannot see. `msinf`, which holds that model, is not here. There is no network.

## Isolation rule

Your entire working world for this task is the current directory. Do not inspect
or access any path outside it, including `/tmp`, `/home`, sibling sandboxes,
previous runs, repository history, or other proposals. Do not search for or read
files outside the current directory. Everything you may use about earlier
rounds is already in `feedback/`, if it is there at all.

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
#: be scored on, the instruction. The instruction used to end "when `proposal.py`
#: is written, you are done", and an agent with four hours took it at its word:
#: observed stopping after 58 seconds, gate check never run.
TASK = """\
Here is the current model: `model.py`.

Rewritable regions: {names}.

Results for this model: see `BRIEF.md`. The training rows are described in
`DATA.md` and loaded by `data/load.py`.{feedback}

Propose an improved version of the rewritable regions, and write them to
`proposal.py` as described. Write a first version early, then keep improving
it: test your ideas against the training rows, run `./.venv/bin/python check.py`,
and stop when more work would no longer make it better -- not when the file
first exists. Only its last version is taken.\
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

#: the gate checker that lands in the sandbox as `check.py`. The runner calls
#: `gates.check` on the age grid and the demography of `Budget.gate_site`; with
#: no mortality table every site's demography is the same, and with one the
#: sandbox's `prop` and `re` rows are that table's, so either way this is the
#: runner's call with the runner's arguments. Sentinels, as in `ROLE`.
CHECK = '''\
"""Run the gates a candidate must pass before it is fitted.

    ./.venv/bin/python check.py              # proposal.py, spliced onto model.py
    ./.venv/bin/python check.py other.py     # any file in proposal.py's format

The same `gates.check` the harness runs, at f64 as the fit runs, on the same age
grid and demography. A pass here is a pass there; it says nothing about how the
candidate will score.
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import gates  # noqa: E402
from discovery.model import Model, parse_blocks  # noqa: E402

HERE = Path(__file__).resolve().parent

proposal = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "proposal.py"
bodies = parse_blocks(proposal.read_text())
if not bodies:
    sys.exit(f"{proposal.name} holds no @discovery blocks")
candidate = Model((HERE / "model.py").read_text()).substitute(bodies)
namespace = {"__name__": "candidate"}
exec(compile(candidate.source, "<candidate>", "exec"), namespace)

# jax arrays, as the runner hands them over; the gates make their own numpy copies
arrays = np.load(HERE / "data" / "arrays.npz")
reasons = gates.check(
    namespace["parameters"],
    namespace["curves"],
    ages=jnp.asarray(arrays["ages"]),
    prop=jnp.asarray(arrays["prop"][@@SITE@@]),
    mu=jnp.asarray(arrays["re"][@@SITE@@] - arrays["grid_ageing"]),
    draws=@@DRAWS@@,
    seed=@@SEED@@,
    inc_max=@@INC_MAX@@,
    namespace=namespace,
)
if reasons:
    print(f"FAILS: {proposal.name} would not be fitted.")
    for reason in reasons:
        print(f"  - {reason}")
    sys.exit(1)
print(f"passes every gate: {proposal.name} would be fitted")
'''


def check_script() -> str:
    """`CHECK` with the runner's own gate settings written into it."""
    budget = Budget()
    return (
        CHECK.replace("@@SITE@@", str(budget.gate_site))
        .replace("@@DRAWS@@", str(budget.gate_draws))
        .replace("@@SEED@@", str(budget.seed))
        .replace("@@INC_MAX@@", repr(budget.inc_max))
    )


def sandboxes(root: Path, agents: int) -> list[Path]:
    """Where each agent works, in agent order."""
    return [root / f"a{agent:02d}" for agent in range(agents)]


def venv(path: Path) -> None:
    """Give one sandbox its own python: `BENCH_PACKAGES`, dmeq, the gates, and nothing else.

    Per agent rather than one shared between them: an agent has a shell, so a
    shared environment is a channel between agents -- one that installs a package
    or drops a file in `site-packages` changes what the others see. `uv` hardlinks
    from its cache, so eight of these cost about as much as one. Pinned to the
    versions this project runs, so the gates agree with the runner's to the bit.
    """
    target = path / VENV
    pins = [f"{name}=={installed(name)}" for name in BENCH_PACKAGES]
    stamp = target / "bench.txt"
    if stamp.is_file() and stamp.read_text().split() == pins:
        return
    shutil.rmtree(target, ignore_errors=True)
    subprocess.run(["uv", "venv", "--quiet", "--python", sys.executable, str(target)], check=True)
    subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(target / "bin" / "python"), *pins], check=True)
    site = next(iter(target.glob("lib/python*/site-packages")))
    for name in BENCH_MODULES:
        module = __import__(name)
        shutil.copytree(Path(module.__file__).parent, site / name, dirs_exist_ok=True)
    for relative, source in BENCH_FILES.items():
        (site / relative).parent.mkdir(parents=True, exist_ok=True)
        (site / relative).write_text(source.read_text() if source else "")
    stamp.write_text("\n".join(pins) + "\n")


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


def repository_model(settings: argparse.Namespace) -> tuple[str | None, dict]:
    """The server url and the model entry the repository's `opencode.json` names.

    `(None, {})` when there is no such file, or `--opencode-config ''`.
    """
    path = Path(settings.opencode_config) if settings.opencode_config else None
    if path is None or not path.is_file():
        return None, {}
    config = json.loads(path.read_text())
    provider_id, _, model_id = (config.get("model") or "").partition("/")
    provider = (config.get("providers") or {}).get(provider_id) or {}
    return (provider.get("settings") or {}).get("baseURL"), (provider.get("models") or {}).get(model_id) or {}


def local(settings: argparse.Namespace) -> None:
    """Fill in what an OpenCode round needs to know about the server it draws on.

    Done before any sandbox is built, so a server that is down or refuses the
    key stops the run here rather than as N agents failing one by one. What the
    flags leave unsaid comes from the server first -- it knows its own slots --
    and then from `opencode.json`.
    """
    url, entry = repository_model(settings)
    settings.url = settings.url or url
    if not settings.url:
        raise SystemExit("--cli opencode needs --url, $DISCOVERY_URL or a baseURL in opencode.json")
    key = os.environ.get(settings.api_key_env) if settings.api_key_env else None
    if settings.api_key_env and not key:
        raise SystemExit(f"${settings.api_key_env} is not set; --api-key-env '' for a server without a key")
    model, context, slots = served(settings.url, key)
    settings.model = settings.model or model
    limit = entry.get("limit") or {}
    settings.context_limit = settings.context_limit or context or limit.get("context")
    settings.output_limit = settings.output_limit or limit.get("output")
    if not settings.context_limit:
        raise SystemExit(f"{settings.url} does not report a context length; pass --context-limit")
    variants = [variant.get("id") for variant in entry.get("variants") or []]
    if settings.variant and variants and settings.variant not in variants:
        raise SystemExit(f"--variant {settings.variant} is not one of opencode.json's {variants}")
    print(
        f"server  {settings.url}: {model}, {settings.context_limit} tokens per request"
        + (f", variant {settings.variant}" if settings.variant else "")
    )
    if slots and slots < settings.concurrency:
        # Not refused, because a queue is not wrong -- but the agents are told a
        # wall-clock budget, and time spent waiting for a slot comes out of it.
        print(
            f"        {slots} slot(s) for {settings.concurrency} agents at once: "
            f"they will queue inside their {settings.session_timeout / 60:.0f}min budget"
        )


def served(url: str, key: str | None) -> tuple[str, int | None, int | None]:
    """The model a local server offers, the context one request may use, and its slots.

    The context is per request rather than per server: llama.cpp divides its
    `n_ctx` between parallel slots and reports one slot's share in `/props`,
    while vLLM reports `max_model_len` on the model card. Neither is assumed --
    a server that reports neither leaves `--context-limit` to be given.
    """
    offered = (fetch(f"{url.rstrip('/')}/models", key) or {}).get("data") or []
    if not offered:
        raise SystemExit(f"{url} serves no models, or did not accept the key")
    card = offered[0]
    context = card.get("max_model_len") or (card.get("meta") or {}).get("n_ctx")
    # `/props` lives beside `/v1` rather than under it, and only llama.cpp has it
    props = fetch(f"{url.rstrip('/').removesuffix('/v1')}/props", key) or {}
    slot = (props.get("default_generation_settings") or {}).get("n_ctx")
    return card["id"], slot or context, props.get("total_slots")


def fetch(url: str, key: str | None) -> dict | None:
    """One GET against the server as json, or `None` if it has no such route."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def opencode_config(settings: argparse.Namespace) -> dict:
    """The one OpenCode configuration an agent reads, in the V2 shape.

    The model entry -- its name, capabilities and reasoning variants -- is the
    repository's `opencode.json`'s, with the limits this server actually has. The
    model's `capabilities` and `limit` are stated rather than left to
    OpenCode's fallbacks for an uncatalogued model -- 200k context, 32k output,
    image input -- which describe no local server. `reasoningField` is where
    llama.cpp streams a reasoning model's thinking; without it the thinking is
    neither recorded nor sent back on the next turn.

    `headerTimeout` is the wait for a response to start, five minutes by
    default, and a server with fewer slots than agents queues requests for
    longer than that. It is bounded by the session instead, which already kills
    an agent that waits too long.
    """
    _, entry = repository_model(settings)
    model: dict = (
        {"name": settings.model, "capabilities": {"tools": True, "input": ["text"], "output": ["text"]}}
        | {key: value for key, value in entry.items() if key != "limit"}
        | {"limit": {"context": settings.context_limit, "output": settings.output_limit or OUTPUT_LIMIT}}
    )
    if settings.reasoning_field:
        model["compatibility"] = {"reasoningField": settings.reasoning_field}
    provider: dict = {
        "name": "Local OpenAI-compatible server",
        "package": "@opencode/ai/providers/openai-compatible",
        "settings": {"baseURL": settings.url, "headerTimeout": int(settings.session_timeout * 1000)},
        "models": {settings.model: model},
    }
    if settings.api_key_env:
        provider["env"] = [settings.api_key_env]
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"{OPENCODE_PROVIDER}/{settings.model}",
        "update": "disable",
        "share": "disabled",
        "snapshots": False,
        "websearch": False,
        "providers": {OPENCODE_PROVIDER: provider},
        "permissions": OPENCODE_PERMISSIONS,
    }


def opencode_home(settings: argparse.Namespace, path: Path) -> Path:
    """Write one agent's OpenCode home beside its sandbox, and return it.

    Beside rather than inside, like `CLAUDE_CONFIG_DIR`: the agent should not
    find its own configuration, or its role, by listing its working directory.
    """
    home = path.parent / f"cfg-{path.name}"
    config = home / ".config" / "opencode"
    config.mkdir(parents=True, exist_ok=True)
    (config / "opencode.json").write_text(json.dumps(opencode_config(settings), indent=2) + "\n")
    (config / "AGENTS.md").write_text(role(settings) + "\n")
    return home


def history(settings: argparse.Namespace) -> Path | None:
    """The feedback pack from every round `--history` names, written beside this round.

    Once per round rather than per sandbox, and kept in `<out>/feedback` -- what
    the agents were shown is part of the record of what they did.
    """
    if not settings.history:
        return None
    pack = feedback.write(
        settings.history,
        settings.out / "feedback",
        split=data.label(settings.splits[0], settings.holdout_seed, settings.n_studies),
        top=settings.feedback_top,
        table=Path(settings.table),
    )
    print(f"feedback {pack}" if pack else "feedback none yet: no round under --history has been fitted")
    return pack


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
    pack = history(settings)
    note = "\n\nWhat earlier rounds found is in `feedback/FEEDBACK.md`: read it first." if pack else ""

    root = settings.sandbox
    if root.exists() and settings.clean:
        shutil.rmtree(root)
    made: list[Path] = []
    for path in sandboxes(root, settings.agents):
        (path / "data").mkdir(parents=True, exist_ok=True)
        (path / "model.py").write_text(parent.source)
        (path / "BRIEF.md").write_text(f"# What this model will be scored on\n\n{text}")
        (path / "DATA.md").write_text(DATA_DOC.format(description=setup.DESCRIPTION))
        (path / "PROMPT.md").write_text(TASK.format(names=names, feedback=note) + "\n")
        (path / "ROLE.md").write_text(role(settings) + "\n")
        built["train"].to_csv(path / "data" / "train.csv", index=False)
        np.savez(path / "data" / "arrays.npz", **built["npz"])
        (path / "data" / "load.py").write_text(LOADER)
        (path / "check.py").write_text(check_script())
        shutil.rmtree(path / "feedback", ignore_errors=True)
        if pack is not None:
            shutil.copytree(pack, path / "feedback")
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
    if settings.cli == "codex":
        effort = (
            ["-c", f"model_reasoning_effort={json.dumps(settings.reasoning_effort)}"]
            if settings.reasoning_effort
            else []
        )
        return [
            settings.cli,
            "exec",
            "--model",
            settings.model,
            *effort,
            "--cd",
            str(path),
            # The outer experiment environment provides process containment;
            # keep Codex unrestricted so it can execute the copied venv. The
            # no-cross-sandbox rule is explicit in ROLE.md and audited below.
            "--sandbox",
            "danger-full-access",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--json",
        ]
    if settings.cli == "opencode":
        # The prompt is the message argument, which is how `opencode run` is
        # documented to take one. The model is passed although the config names
        # it too, so the argv says what was drawn without the config beside it.
        variant = f"#{settings.variant}" if settings.variant else ""
        return [
            settings.cli,
            "run",
            "--standalone",
            "--format",
            "json",
            "--auto",
            "--thinking",
            "--model",
            f"{OPENCODE_PROVIDER}/{settings.model}{variant}",
            (path / "PROMPT.md").read_text(),
        ]
    argv = [
        settings.cli,
        "-p",
        "--bare" if settings.auth == "api-key" else "--safe-mode",
        "--model",
        settings.model,
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--tools",
        TOOLS,
        "--append-system-prompt",
        (path / "ROLE.md").read_text(),
        "--exclude-dynamic-system-prompt-sections",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--output-format",
        "stream-json",
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
    # The subprocess is started in the sandbox, but `PWD` and `OLDPWD` would
    # still name the directory this runner was launched from -- the project, in
    # an environment any `env` prints. OpenCode goes further and takes its
    # workspace from `PWD`: observed listing and writing into the repository
    # while its process sat in the sandbox.
    env.pop("OLDPWD", None)
    env["PWD"] = str(path)
    if settings.cli == "codex":
        return {k: v for k, v in env.items() if k not in {"OPENAI_API_KEY", "CODEX_API_KEY"}}
    if settings.cli == "opencode":
        home = opencode_home(settings, path)
        return env | {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "XDG_STATE_HOME": str(home / ".local" / "state"),
            # the cache holds the binaries OpenCode fetches for its tools, not
            # anything about a session, so it stays shared rather than being
            # fetched again by every agent
            "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")),
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
        }
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
    prompt = stdin(settings, path)
    attempts = 1 + (OPENCODE_RETRIES if settings.cli == "opencode" else 0)
    started = time.perf_counter()
    log = out / f"a{agent:02d}.jsonl"
    for attempt in range(1, attempts + 1):
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
                    process.communicate(prompt),
                    timeout=settings.session_timeout - (time.perf_counter() - started),
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "agent": agent,
                    "returncode": None,
                    "error": f"the session did not finish within {settings.session_timeout:.0f}s",
                    "seconds": time.perf_counter() - started,
                    "attempts": attempt,
                    "log": str(log),
                }
        if process.returncode == 0 or stepped(log):
            break
    return {
        "agent": agent,
        "returncode": process.returncode,
        "error": None if process.returncode == 0 else (stderr or b"").decode()[-3000:] or log.read_text()[-3000:],
        "seconds": time.perf_counter() - started,
        "attempts": attempt,
        "log": str(log),
    }


def stdin(settings: argparse.Namespace, path: Path) -> bytes:
    """What an agent is sent on stdin.

    OpenCode has its prompt in its argv and its role in `AGENTS.md`, and Claude
    has its role in `--append-system-prompt`. Codex has no flag for a role, and
    `ROLE.md` in the sandbox is read only by an agent that thinks to -- observed
    not to -- so its role goes ahead of the task: the budget, the isolation rule
    and `check.py` are not things to be found by chance.
    """
    if settings.cli == "opencode":
        return b""
    task = (path / "PROMPT.md").read_bytes()
    if settings.cli == "codex":
        return (path / "ROLE.md").read_bytes() + b"\n---\n\n" + task
    return task


def stepped(log: Path) -> bool:
    """Whether an OpenCode session got as far as asking the model anything.

    One that did not has left the sandbox as it found it, which is what makes
    starting it again the same draw.
    """
    return any(event.get("type") == "step_start" for event in events(log))


def events(log: Path) -> list[dict]:
    """A session's stream, one json object per line, skipping what is not one."""
    found = []
    if not log.exists():
        return found
    for line in log.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            found.append(event)
    return found


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
        if "sessionID" in event:
            opencode(event, turns, usage)
            continue
        if kind == "error":
            turns.append({"role": "system", "content": json.dumps(event, default=str)})
            continue
        if kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                turns.append({"role": "assistant", "content": item.get("text", "")})
            elif item.get("type") in {"command_execution", "file_change"}:
                turns.append({"role": "assistant", "content": json.dumps(item, default=str)})
            continue
        if kind == "turn.completed":
            usage = event.get("usage") or {}
            if event.get("model"):
                usage["model"] = event["model"]
            continue
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
                "web_searches": sum((m or {}).get("webSearchRequests", 0) for m in per_model.values()),
            }
    if resolved:
        usage["model"] = resolved
    return turns, usage


def opencode(event: dict, turns: list[dict], usage: dict) -> None:
    """Read one event of `opencode run --format json` into the turns and usage.

    A tool call and its result arrive as one `tool_use` event, completed or
    failed, and are split into the two turns the other harnesses produce. The
    token counts summed from `step_finish` are a floor rather than a total: the
    last step's event is not always emitted before the process exits, and
    `opencode_usage` replaces them with the session's own record where it can.
    """
    kind = event.get("type")
    part = event.get("part") or {}
    usage.setdefault("session_id", event["sessionID"])
    if kind == "text":
        turns.append({"role": "assistant", "content": part.get("text", "")})
    elif kind == "reasoning":
        turns.append({"role": "assistant", "content": f"[thinking]\n{part.get('text', '')}"})
    elif kind == "tool_use":
        state = part.get("state") or {}
        call = json.dumps(state.get("input", {}), default=str)
        turns.append({"role": "assistant", "content": f"[tool_use {part.get('tool')}]\n{call}"})
        result = state.get("output") if state.get("status") == "completed" else state.get("error")
        turns.append({"role": "user", "content": f"[tool_result {state.get('status')}]\n{result}"})
    elif kind == "step_finish":
        tokens = part.get("tokens") or {}
        cache = tokens.get("cache") or {}
        for key, value in (
            ("input_tokens", tokens.get("input")),
            ("output_tokens", tokens.get("output")),
            ("reasoning_tokens", tokens.get("reasoning")),
            ("cache_read_input_tokens", cache.get("read")),
            ("cache_creation_input_tokens", cache.get("write")),
        ):
            usage[key] = usage.get(key, 0) + (value or 0)
        usage["num_turns"] = usage.get("num_turns", 0) + 1
        usage["usage_source"] = "stream"
    elif kind == "error":
        turns.append({"role": "system", "content": json.dumps(event.get("error"), default=str)})


def opencode_usage(path: Path, session: str | None) -> dict:
    """What one OpenCode session cost and which model served it, from its database.

    The database is the agent's own, under the `XDG_DATA_HOME` beside its
    sandbox, so the session is the only one in it that matters. `session_v2` is
    OpenCode's internal table rather than a documented interface, so anything
    unexpected here leaves the stream's floor in place instead of failing the
    round.
    """
    database = path.parent / f"cfg-{path.name}" / ".local" / "share" / "opencode" / "opencode.db"
    if session is None or not database.exists():
        return {}
    try:
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as db:
            row = db.execute(
                "select model, cost, tokens_input, tokens_output, tokens_reasoning, "
                "tokens_cache_read, tokens_cache_write from session_v2 where id = ?",
                (session,),
            ).fetchone()
        model = json.loads(row[0]).get("id") if row and row[0] else None
    except (sqlite3.Error, ValueError, AttributeError):
        return {}
    if row is None:
        return {}
    found = {
        "total_cost_usd": row[1],
        "input_tokens": row[2],
        "output_tokens": row[3],
        "reasoning_tokens": row[4],
        "cache_read_input_tokens": row[5],
        "cache_creation_input_tokens": row[6],
        "usage_source": "opencode.db",
    }
    if model:
        found["model"] = model
    return found


def outside_workspace(turns: list[dict], path: Path) -> list[str]:
    """Return transcript lines that name paths outside this agent's sandbox."""
    root = str(path.resolve())
    suspicious: list[str] = []
    for turn in turns:
        content = str(turn.get("content", ""))
        for line in content.splitlines():
            if "/tmp/" in line or "/home/" in line:
                if root not in line:
                    suspicious.append(line[:1000])
    return suspicious


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
        if settings.cli == "opencode":
            usage |= opencode_usage(path, usage.get("session_id"))
        outside = outside_workspace(turns, path)
        if outside:
            usage["outside_workspace"] = outside
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
                "outside_workspace": outside or None,
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
                "harness": HARNESSES.get(settings.cli, "claude-code"),
                # `message.model` comes back as the alias itself -- a first-party
                # session reports `claude-opus-5`, not a dated id -- so what
                # actually pins this round is the alias, the CLI version and the
                # day it was drawn, together.
                "drawn_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "cli_version": version(settings.cli),
                "tools": TOOLS,
                "auth": {"codex": "subscription", "opencode": "server"}.get(settings.cli, settings.auth),
                # where a local server was reached and what it was told it could
                # hold, since neither is in the model id
                "server": settings.url if settings.cli == "opencode" else None,
                "context_limit": settings.context_limit if settings.cli == "opencode" else None,
                "variant": settings.variant if settings.cli == "opencode" else None,
                "reasoning_effort": settings.reasoning_effort if settings.cli == "codex" else None,
                "session_timeout": settings.session_timeout,
                # the rounds the agents' feedback/ was built from; `<out>/feedback`
                # is the pack itself
                "history": [str(path) for path in settings.history],
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
        return (
            subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        return None


def options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="examples.malaria.agentic", description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs_agentic/01-claude/curves-griffin"),
        help="two levels under the sweep root, so sweep.py finds it",
    )
    parser.add_argument("--agents", type=int, default=8, help="proposals drawn, one agent each")
    parser.add_argument("--target", default="curves", choices=TARGETS)
    parser.add_argument("--start", default="griffin", choices=START_NAMES)

    harness = parser.add_argument_group("coding harness")
    harness.add_argument("--cli", default="claude")
    harness.add_argument(
        "--model",
        default=None,
        help="pinned alias or full id; the id it resolves to is what round.json records. "
        f"Defaults to {DEFAULT_MODELS}, or for opencode to whatever the server serves",
    )
    harness.add_argument(
        "--auth",
        default="api-key" if os.environ.get("ANTHROPIC_API_KEY") else "session",
        choices=("api-key", "session"),
        help="api-key uses --bare (stronger isolation); session uses --safe-mode",
    )
    harness.add_argument(
        "--sandbox",
        type=Path,
        default=None,
        help=f"neutral root for the agents' directories (default under {SANDBOX_ROOT})",
    )
    harness.add_argument("--concurrency", type=int, default=4, help="agents running at once")
    harness.add_argument(
        "--session-timeout",
        type=float,
        default=None,
        help=f"seconds one agent may take, and what it is told; defaults to {SESSION_TIMEOUTS}",
    )
    harness.add_argument(
        "--reasoning-effort", default="high", help="codex only: model_reasoning_effort; '' for the model's default"
    )
    harness.add_argument("--max-budget-usd", type=float, default=5.0, help="0 to leave it unbounded")
    harness.add_argument("--keep", action="store_true", help="do not clear the sandbox root first")

    local = parser.add_argument_group("local server, --cli opencode only")
    local.add_argument(
        "--url",
        default=os.environ.get("DISCOVERY_URL"),
        help="OpenAI-compatible base url, e.g. http://127.0.0.1:8001/v1 [$DISCOVERY_URL]",
    )
    local.add_argument(
        "--api-key-env",
        default="LLM_API_KEY",
        help="the variable holding the server's key; '' for a server without one",
    )
    local.add_argument(
        "--context-limit",
        type=int,
        default=None,
        help="tokens one request may use; defaults to what the server reports per slot",
    )
    local.add_argument(
        "--output-limit",
        type=int,
        default=None,
        help=f"tokens one reply may use; defaults to opencode.json's, else {OUTPUT_LIMIT}",
    )
    local.add_argument(
        "--opencode-config",
        default=str(OPENCODE_CONFIG),
        help="the opencode.json each agent's config is built from; '' for none",
    )
    local.add_argument(
        "--variant", default="max", help="the opencode.json reasoning variant to run with; '' for the model's default"
    )
    local.add_argument(
        "--reasoning-field",
        default="reasoning_content",
        help="the message field a reasoning model streams its thinking in; '' for none",
    )

    inputs = parser.add_argument_group("what the brief describes")
    inputs.add_argument("--table", default=str(data.DEFAULT_TABLE))
    inputs.add_argument("--eir-priors", default=str(data.DEFAULT_EIR_PRIORS))
    inputs.add_argument("--mortality", default=data.DEFAULT_MORTALITY)
    inputs.add_argument("--splits", default=",".join(data.SPLITS))
    inputs.add_argument("--holdout-seed", type=int, default=42)
    inputs.add_argument("--n-studies", type=int, default=3)
    inputs.add_argument("--warmup", type=int, default=Budget.warmup)
    inputs.add_argument("--samples", type=int, default=Budget.samples)
    inputs.add_argument("--chains", type=int, default=Budget.chains)

    previous = parser.add_argument_group("what earlier rounds found")
    previous.add_argument(
        "--history",
        type=Path,
        nargs="*",
        default=[],
        help="fitted rounds, or directories above them, to build feedback/ from. "
        "Training-fold numbers only; rounds fitted under another split are skipped",
    )
    previous.add_argument(
        "--feedback-top", type=int, default=8, help="candidates given a directory of their own in feedback/"
    )

    phases = parser.add_argument_group("phases")
    phases.add_argument("--build-only", action="store_true", help="write the sandboxes and stop")
    phases.add_argument(
        "--collect-only", action="store_true", help="validate sandboxes an earlier run left, without running anything"
    )

    settings = parser.parse_args(argv)
    settings.splits = tuple(s.strip() for s in settings.splits.split(",") if s.strip())
    if unknown := set(settings.splits) - set(data.SPLITS):
        raise SystemExit(f"unknown split(s) {sorted(unknown)}; expected some of {list(data.SPLITS)}")
    if settings.model is None and settings.cli != "opencode":
        settings.model = DEFAULT_MODELS.get(settings.cli, DEFAULT_MODELS["claude"])
    if settings.session_timeout is None:
        settings.session_timeout = SESSION_TIMEOUTS.get(settings.cli, SESSION_TIMEOUTS["claude"])
    if settings.sandbox is None:
        settings.sandbox = SANDBOX_ROOT / f"{settings.target[:1]}{settings.start[:1]}-{os.getpid():05d}"
    settings.clean = not settings.keep
    return settings


async def main(argv: list[str] | None = None) -> int:
    settings = options(argv)
    if settings.cli == "claude" and settings.auth == "api-key" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("--auth api-key needs ANTHROPIC_API_KEY; use --auth session to use the normal login")
    if shutil.which(settings.cli) is None:
        raise SystemExit(f"no {settings.cli!r} on PATH")
    if settings.cli == "opencode" and not settings.collect_only:
        local(settings)

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
    print(f"fit it here:    uv run python -m examples.malaria --proposals {out / 'proposals'} --out {out}")
    print(f"or on Slurm:    examples/malaria/slurm/remote.sh submit {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
