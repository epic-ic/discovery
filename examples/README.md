# Examples

Runnable end-to-end uses of the harness. They need their own dependencies:

```sh
uv sync --group examples
```

## birthdates

Twenty years of daily birth counts, 1969-1988, and a search for the process
behind them.

```sh
uv run python -m examples.birthdates --url http://host:8000/v1 --agents 8
```

`--url` is any OpenAI-compatible endpoint, or `$DISCOVERY_URL`; the model
defaults to whatever that server says it is serving. `--no-search` skips the
language model entirely and fits only the starting model, which is the quickest
way to see the evaluation path work.

### What is held fixed

The candidate is a numpyro program with one rewritable region, `process`, which
computes `log_rate` for each day from the day's features -- time, day of year,
day of week, month, year. Everything after that region is scaffolding the
proposer may not touch, and the last of it is the observational likelihood:

```python
concentration = sample("concentration", dist.Exponential(0.01))
sample("births", dist.NegativeBinomial2(jnp.exp(log_rate), concentration), obs=births)
```

Holding the observation model still is what makes the scores comparable. Every
candidate is a different story about how the rate varies over time and the same
story about how a count is drawn given the rate, so a better held-out likelihood
is a better account of the generating process, and not a noise model widened
until the residuals fit inside it.

The starting process is a level and a linear trend, which is wrong in every way
the data are interesting: no yearly season, no weekday effect, no holidays.

### Looking at the data first

```sh
uv run python -m examples.birthdates --url http://host:8000/v1 --agents 8 --analysis
```

With `--analysis`, each agent gets its own conversation and a `run_python` tool
over the training days -- numpy, pandas, `births` and the five features, and a
`df` of all of them -- and may run up to `--tool-budget` snippets before it
proposes. `examples/birthdates/analysis.py` is what builds that namespace.

It builds it from `data.load(...)[0]`, the training split, and never touches the
second half of the pair. A proposer that could measure the years it is about to
be scored on would be choosing its model with the answer in front of it, and
`elpd_holdout` would stop meaning what the table says it means. The rest of the
environment's isolation is described in the top-level README; the split is the
part that is this example's responsibility.

The server has to be started for tool calling -- for vLLM that is
`--enable-auto-tool-choice --tool-call-parser hermes` for the Qwen models.
Without it every agent fails with a `ToolCallError` saying so.

`--analysis-workers` (4) bounds how many snippets run at once across all agents,
whatever `--agents` is. Nothing is logged unless the queue makes an agent wait,
which is the signal that the number wants raising.

### The partition

Train on 1969-1984 (5844 days), hold out 1985-1988 (1461 days). The holdout is
the tail of the series rather than a random sample of days, and `t` keeps
counting across the split, so a candidate has to extrapolate its trend into
years it has never seen. A random holdout would let a model that merely smooths
its neighbours score as well as one that found the process.

### The fit

Every candidate gets the same NUTS budget -- 4 chains, 500 warmup, 500 draws --
because a model that only looks better for having been sampled harder has not
been found to be better. Each fit runs in its own subprocess: `discovery` has no
sandbox yet, and model source written by a language model is exactly what you do
not exec in the process holding your results. A fit that raises, hangs, or is
killed on `--fit-timeout` becomes one failed `Result` and costs the batch
nothing else.

One fit produces everything, in one `Posterior.score` call, because a diagnostic
recomputed from a second fit would describe a different fit:

| metric | what it answers |
| --- | --- |
| `kl.<parameter>`, `kl_min`, `unidentified` | is every parameter identified? The information the data added to each parameter's prior, in nats: a k-nearest-neighbour estimate of KL(posterior ‖ prior) per coordinate, in unconstrained space. Zero means the data never moved it. |
| `elpd_holdout`, `elpd_holdout_per_day`, `elpd_train` | the model's likelihood given data, log pointwise predictive density over the posterior draws. |
| `coverage95`, `coverage_error` | how often the held-out day fell inside the 95% posterior predictive interval. |
| `rhat.<parameter>`, `rhat_max` | rank-normalised split R-hat, per parameter. |
| `ess_tail.<parameter>`, `ess_tail_min` | tail effective sample size, per parameter. |
| `mcmc_seconds`, `total_seconds`, `parameters` | what the fit cost. `mcmc_seconds` includes jit compilation. |
| `divergences`, `pit_ks`, `nonfinite` | the rest of the warnings. |

Per-parameter numbers are named one by one (`rhat.trend`, `kl.seasonal[3]`)
rather than packed into a vector, because candidates do not share a parameter
list. A metric that is undefined -- R-hat for a parameter that never moved --
comes back as `NaN` and is counted in `nonfinite`; the summaries are NaN-aware.

Two diagnostics travel with the numbers, written to `artifacts/` beside the
report:

* **ppc** -- the held-out days against the posterior predictive median and 95%
  band, whole holdout and first year.
* **calibration** -- the empirical CDF of the randomised PIT against uniform,
  with a Kolmogorov-Smirnov band. Randomised because the observation model is
  discrete, and estimated from the predictive draws rather than a CDF so it
  stays correct whatever the fixed observation model is.

### Reading a run

```
  #   elpd/day cover95  kl_min unid  rhat ess_tail   div   secs pars
  4     -8.133   0.950    0.49    0  1.04      164     0  162.9   22
  2     -8.182   0.938    0.02    1  1.02      214     0  236.4   12
  0*    -8.653   0.956    3.64    0  1.01      291     0   35.3    3
  1  IndentationError: unexpected indent
```

Ranked by held-out likelihood; `0*` is the model the search started from, fitted
alongside the proposals so the run says whether the search actually beat it. A
proposal that would not parse, would not run, or would not converge stays in the
report with its error: the models worth looking at are often the ones that
failed.

Everything, including the model source, is in `runs/birthdates/report.jsonl`.
