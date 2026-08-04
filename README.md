# Discovery

A package for model discovery. It orchestrates agents for a model search.

# Installation

```sh
uv sync
```

# Getting started

To get started specify:

 * an initial model,
 * an OpenAI compatible url for LLM completions,
 * a code execution sandbox,
 * an Evaluator,
 * a Search, and
 * optional Diagnostics

# Roadmap

 - [x] Model proposer
   - Stores model url and transcript
   - Creates prompt

A model is a program. Regions the search may rewrite are delimited by marker
comments; everything else is fixed scaffolding that the proposer may not touch:

```python
def dv_dt(x, t, v, params):
    # @discovery:begin dynamics
    return params["F0"] * np.sin(t) - params["beta"] * v
    # @discovery:end
```

The proposer sends the program and any feedback, then splices the returned
blocks back in by name:

```python
proposer = ModelProposer(url="http://host:8000/v1", llm_model="Qwen/Qwen3.5-35B-A3B-FP8")
proposed = await proposer.propose(Model(source), feedback="log likelihood: -1843.2")
```

The client is asynchronous: a search runs many proposers at once, and they do
nothing but wait on the inference server.
 - [x] I.I.D. Search
   - N agents

`IIDSearch` asks for N proposals against the same parent model in a **single
request**, using `n`. The prompt is prefilled once rather than N times, and the
server draws the samples in one batch:

```python
async with IIDSearch(
    url="http://host:8000/v1",
    llm_model="Qwen/Qwen3.5-35B-A3B-FP8",
    agents=16,
    timeout=300.0,
    decode=Decode(
        temperature=1.0, top_p=0.95, top_k=50,
        max_tokens=4096, thinking_token_budget=512,
    ),
) as search:
    proposals = await search.propose(Model(source), feedback="log likelihood: -1843.2")

for candidate in models(proposals):
    ...
```

`propose` is the only operation, and drawing one batch is the whole algorithm.
Iterating would mean *selecting* among the candidates, which needs a score, so
the outer loop belongs to the Evaluator and to the searches that carry state
between rounds — SMC and islands, below. The conversational machinery those will
want is already in `ModelProposer`; an i.i.d. search simply never uses it.

`Decode` carries the usual batch decode parameters; `top_k`, `min_p`,
`repetition_penalty` and `thinking_token_budget` are passed as vLLM extensions,
and anything else can go in `extra`. A `seed` governs the whole request — vLLM
gives no way to seed one sample within it, and it does not buy replay in any
case: repeating a seeded request returns different completions, because
continuous batching makes the numerics depend on what else is in the batch.

A sample that comes back unreadable becomes a `Proposal` carrying the reason and
the reply behind it, so it costs the batch nothing but itself. A failure of the
request fails the whole batch, since there is only the one. Two things are worth
knowing when running against a reasoning model:

 * **Bound the thinking.** Left alone, a reasoning model can spend an entire
   `max_tokens` before emitting any answer and return empty content. Against
   Qwen3.5-35B-A3B this was every failure we saw: 38% of requests. Setting
   `thinking_token_budget` fixes it — one measured reply needed 1500+ tokens and
   returned nothing, and came back valid in 323 with a 256-token budget.
   `thinking=False` turns reasoning off entirely (47 tokens). The error
   distinguishes a truncated reply from a merely unusable one.
 * **A batch waits for its slowest sample.** Runaway traces, not the number of
   agents, set the wall clock, which is the other reason to bound the thinking.
   `timeout` bounds the request as a whole.
 - [x] Evaluator
   - Score a model

An evaluator answers one question about one model and says so in numbers: a
dictionary of scalars or vectors, named. `Diagnostics` is an evaluator that may
also return an `Artifact` — a posterior predictive plot, a trace — alongside its
numbers, in the same call and the same pass over the model.

```python
class Likelihood(Evaluator):
    async def score(self, model: Model) -> dict:
        return {"loglik": await sandbox.run(model), "folds": [...]}

class Posterior(Diagnostics):
    async def score(self, model: Model) -> dict:
        fit = await sandbox.run(model)
        return {"rhat": fit.rhat, "trace": Artifact("trace", fit.png)}

evaluations = await evaluate(
    models(proposals), [Likelihood(), Posterior()], concurrency=8, timeout=120.0
)
```

The unit of work is one evaluator on one model, and `concurrency` bounds how
many of those are in flight, since the sandbox rather than the event loop is the
scarce thing. One failure costs one result: an evaluator that raises, times out,
or hands back something that is not a number becomes a `Result` carrying the
exception, and the rest of the batch is unaffected. Evaluations keep the order
of the models and results the order of the evaluators, so a report reads the
same way twice.

 - [x] Reporting
   - jsonl of model, scores, diagnostics

```python
write(evaluations, "runs/007/report.jsonl", append=True)
```

One line per model, holding the model source, the metrics grouped by the
evaluator that produced them, and any errors. Jsonl because a record is
naturally nested — a vector is a list — and because a line is complete on its
own: a search can append each round as it finishes, and `read` skips a
truncated tail, so the report of a run that died halfway is still a report.
Images are the one thing that does not fit on a line; they are written to
`artifacts/` beside the report and the record points at them.
 - [x] Add context for proposers
   - Data analysis at proposal time

A proposer that has looked at the data writes a better model. `Analysis` is the
environment it looks with: a pool of worker processes holding numpy, pandas and
the data being fitted, offered to the agent as one tool, `run_python`.

```python
async with Analysis(setup="examples.birthdates.analysis", workers=4) as pool:
    async with AgentSearch(
        url="http://host:8000/v1", llm_model="Qwen/Qwen3.5-35B-A3B-FP8",
        agents=16, analysis=pool, tool_budget=6,
    ) as search:
        proposals = await search.propose(Model(source), feedback=brief)
```

`setup` names an importable module exposing `namespace()`, which returns the
names the snippet runs against, and `DESCRIPTION`, which is what the agent is
told is in it. **`namespace()` is called before every execution**, so each
snippet gets its own copy: mutation is contained by construction, nothing
carries over between calls, and any worker can therefore take any call. That
last point is what makes the pool a queue rather than an assignment — there are
`workers` of them however many agents are asking, and an agent that has to wait
longer than `warn_after` is logged with the number of agents still queued, since
that is the number that says whether `workers` is set right.

**It is not a sandbox.** Three cheap mechanisms cover the failures that actually
happen when a language model explores data:

 * **Nothing is written, anywhere.** Once the imports are done the worker
   installs a `sys.addaudithook` that refuses to open a file for writing, to
   rename or unlink one, to open a socket, or to start a process. Audit hooks
   are stdlib, they see the operation rather than the library that reached for
   it, and they cannot be uninstalled. `RLIMIT_FSIZE` is clamped behind it, but
   the hook is what does the work: an rlimit alone still lets `open(path, "w")`
   create and *truncate* a file, because neither writes a byte.
 * **Workers are disposable.** A snippet still running after `timeout` has its
   worker killed and replaced, and every worker is recycled after `recycle`
   executions — a fresh namespace resets the bindings but not `sys.modules`, and
   recycling bounds how far a snippet that patched something can follow later
   agents.
 * **Failures are in-band.** An exception, a runaway loop, a worker that died:
   all of it comes back to the agent as text, and costs it one of its
   `tool_budget` calls rather than its proposal.

The last result of every turn ends with the budget remaining -- `[3 tool calls
left.]`, and finally `[No tool calls left. Answer now with the fenced code
block.]` -- because an agent cannot count its own calls. Told only the limit up
front it keeps exploring until the harness cuts it off, and everything it
learned is discarded with the proposal it never wrote. Raising `tool_budget`
does not help on its own: measured against Qwen3.5-35B-A3B, going from 4 calls
to 6 doubled the number of agents that exhausted it.

A proposal that fails is not committed to its transcript, so `ProposalError`
carries an `exchange` -- the turns it had got through. Without tools a failed
agent loses one reply and the reply is the whole story; with them it loses every
snippet it ran and everything it read, which is most of why it failed.

None of that survives a determined adversary with a Python prompt. What it stops
is the snippet that overwrites the data file it was reading.

Two things the tool changes about the search. First, a multi-turn exchange
cannot be one `n`-sample request, so `AgentSearch` is N requests where
`IIDSearch` is one, and the prompt is prefilled per agent; prefix caching
recovers most of it, but the batched search remains the cheaper way to draw
proposals that need not look at anything. Second, **the server has to be started
for tool calling** — vLLM only fills `tool_calls` with `--enable-auto-tool-choice
--tool-call-parser <parser>`, and without them the call arrives as text in
`content` and reads exactly like a model that ignored its instructions.
`ToolCallError` says so rather than blaming the reply.

Each agent keeps its whole exchange — every snippet it ran and everything it
read — in its transcript, because that is the evidence behind the model it went
on to write.

 - [ ] Sandbox for model generator
 - [ ] SMC Search
   - N particles
 - [ ] Islands Search
   - N islands
   - N agents

# Examples

`examples/` runs the whole loop against real data; see `examples/README.md`.

```sh
uv sync --group examples
uv run python -m examples.birthdates --url http://host:8000/v1 --agents 8
```

`birthdates` searches for the process behind twenty years of daily birth counts,
holding the observational likelihood fixed outside the rewritable region and
scoring every proposal on one NUTS fit with a constant budget: identifiability,
held-out likelihood, coverage, R-hat, tail-ESS and the cost of the fit, with a
posterior predictive check and a calibration plot. Until the sandbox above
exists, the example carries its own -- each fit is a child process, so a model
that hangs or crashes costs one result.
