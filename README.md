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
 - [ ] Evaluator
   - Score a model
 - [ ] Reporting
   - csv of model, scores, diagnostics
 - [ ] Sandbox for model generator
 - [ ] SMC Search
   - N particles
 - [ ] Islands Search
   - N islands
   - N agents
