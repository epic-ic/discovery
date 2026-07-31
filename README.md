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
proposed = proposer.propose(Model(source), feedback="log likelihood: -1843.2")
```
 - [ ] I.I.D. Search
   - N agents
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
