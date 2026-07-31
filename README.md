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
 - [ ] Sandbox for model generator
 - [ ] SMC Search
   - N particles
 - [ ] Islands Search
   - N islands
   - N agents
