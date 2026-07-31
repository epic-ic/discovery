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

 - [ ] Model proposer
   - Stores model url and transcript
   - Creates prompt
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

# Example for tests

symbols:
[
  "dv_dt",
  "x",
  "t",
  "v"
]

symbol descriptions:
[
  "Acceleration in Nonl-linear Harmonic Oscillator",
  "Position at time t",
  "Time",
  "Velocity at time t"
]

symbol properties (O -> "operator/constant", V -> "variable"):
[
  "O",
  "V",
  "V",
  "V"
]

expression:
F0*sin(t) - beta*sin(v(t)) - omega0**2*x(t)**3 - omega0**2*x(t)*exp(-Abs(x(t)))
