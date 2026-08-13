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

## malaria

Age-stratified malaria incidence and prevalence from African field studies, and
a search for the map from parameters to age curves. `msinf` holds the
observational model fixed and does all the inference; this example owns the
search and the harness contract.

```sh
uv run python -m examples.malaria --url http://host:8000/v1 --agents 8 --target immunity --start griffin
```

### One interpreter, and one thing that is not on this machine

Both projects now run on the same Python -- `msinf` is `>=3.11`, `discovery` is
`>=3.12` (arviz-stats' floor, not the harness's) -- so `msinf` installs into the
same environment and the child process runs under `sys.executable`, exactly as
`birthdates` does:

```sh
uv sync --group examples --group malaria
```

`msinf` is not on PyPI, so the `malaria` group takes it from a sibling checkout
at `../msinf`; it is a group of its own so `examples` stays installable without
one. The subprocess boundary is still there, for the reason it always was: model
source written by a language model does not run in the process holding the
results.

The `<=20y` study table and the site EIR priors are **not in this repository and
not on this machine**. Point the example at them:

```sh
export MALARIA_TABLE=/path/to/study_data_2014_age_le20.csv
export MALARIA_EIR_PRIORS=/path/to/eir_site_priors_griffin2014.csv
```

Without them the run stops on the first line with a `MissingData` naming both
files, rather than failing inside a child process forty minutes later.

### What is held fixed

Everything except the curve provider. The study random effects (`u` incidence,
`w` prevalence, Gauss-Hermite integrated) and the site-age effect `v_jk`
(analytically integrated, negative binomial), the incidence Poisson and the
prevalence beta-binomial, the site EIR priors, `ft_study`, the age grid, the
demography and the age-bin averaging are all `msinf`'s and none of them is
rewritable.

Two rewritable regions rather than birthdates' one, and the split is forced
rather than chosen: NumPyro sample sites cannot be created inside the `vmap`
over sites, so a candidate declares its parameters in one block and evaluates
curves in another.

```python
# @discovery:begin parameters   sample statements, once per evaluation
# @discovery:begin curves       the whole mechanism, at --target curves
# @discovery:begin immunity     the immunity model, at --target immunity
```

The block returns prevalence in `[0, 1]` and incidence in episodes per
person-**year**. A fixed `curves` below it converts to what the likelihood
actually consumes -- a population fraction per person-**day**, since `msinf`
averages a bin as `sum_a w(a) curve(a) / sum_a w(a) prop(a)` and scales incidence
by 365 -- so a candidate is written in the units it can reason about.

### Two targets, one model

`--target` chooses how much of one model the markers open up, not which model:

* `curves` -- the immunity model, the equilibrium state solve and the driver are
  all inside the block. A candidate replaces the equilibrium model entirely.
* `immunity` -- only the `eps -> (foi, phi, q, b)` block is inside. The state
  solve and the driver sit below it as fixed text, which a candidate can read but
  not change: what *consumes* the four values it returns should not have to be
  guessed at.

Both assembled modules define exactly the same things, so the incumbent's score is
one number rather than two, and it is the reference either search is judged
against.

### Two starting models

One per run -- `--start scratch` or `--start griffin`. Comparing them means two
runs and two reports, which is the shell's job: a single run that drew from both
would double its own length and the two rounds would still have to stay blind to
each other, since they stop being that comparison the moment either can read the
other's proposals.

* `scratch` -- prevalence and incidence that respond to EIR and to nothing else,
  or immunity that does not accumulate. The analogue of birthdates'
  level-and-trend.
* `griffin` -- the incumbent equilibrium model, for *correction* rather than
  discovery. It is misspecified; the question is whether a search seeded with it
  beats one started from nothing, and whether the two converge anywhere near
  each other.

`griffin`'s mechanism is not restated -- it is `inspect.getsource` from `dmeq`, so
what a proposer reads is byte for byte what the fit runs. `tests/test_malaria.py`
still pins it against `msinf.curves.griffin_provider` at a fixed seed and
parameter vector, at both targets, because the parameter block and the 25-line
driver standing in for `dmeq._solve` are the two pieces that are not extracted.

Expect something flexible in age × EIR with no mechanism in it to score near the
top. That is not a failure -- it is the empirical ceiling on held-out
likelihood, and it is the number the mechanistic candidates should be judged
against.

### The split

One: `study`, three whole studies dropped from the fit and scored on. It is the
ranked score, and it is worth being explicit about what it is blunt about. A
held-out study's `log_eir` sits on its prior (sd 0.3-0.8 on the log scale), so a
large part of the number is how a candidate behaves when transmission intensity
is known only to within a wide prior -- scaffolding every candidate shares. At
the default seed the fold is studies {2, 10, 11}: 42 rows, **16 incidence cells
plus 19 prevalence rows = 35 scoring units**. That separates working candidates
from broken ones cleanly (run 04: -193 against -310) and orders the top few much
less confidently -- they sat within 20 nats of each other.

Which is what the table's **`vs best`** column is for: the paired difference
against the top row and its standard error, in nats. Pairing rather than
comparing two totals, because both candidates were scored on the same units and
a unit that is hard for one is hard for both -- differencing removes that shared
difficulty and typically cuts the error to a third. A lead smaller than about
twice its error is not a lead. The per-unit vectors it is computed from are
`<split>.pointwise_holdout_*`, and `estimators.elpd_diff` does the arithmetic.

What no arithmetic over one fold can recover is *which* studies were drawn into
it. That is the larger uncertainty here and it takes more folds, not a better
error bar -- see the note on leave-one-study-out below.

`age_band` used to be here, and was the ranked score. It is a better question --
it leaves `log_eir` informed and so tests age structure rather than transmission
-- but not one this table can be asked. An incidence cell is a
(site, age_bracket) pair and 43 of the 76 cells span more than one `age0`, so
nearly every boundary cuts a cell in half and `msinf.splits` refuses it. The only
legal interior cuts are 2y, which holds out 71% of the rows and leaves the fit
training on infants alone, and 15y, which holds out 19 rows. The 5y this example
defaulted to was refused outright. `msinf` still supports the kind for its own
experiments; this example no longer offers it.

The split is not drawn here. Which rows can be scored apart from which is a
property of the likelihood: no incidence cell may straddle the boundary, because
`v_jk` is integrated out at the cell level and the marginal is a joint density
over the whole cell; and site indices renumber the moment a study is dropped.
`data.py` names a split, `msinf` draws it and refuses an incoherent one.

**Not done yet: leave-one-study-out.** One fold of three studies out of fourteen
is one draw, and searching many candidates against it eventually fits the
idiosyncrasies of studies {2, 10, 11}. The fix is more folds rather than a bigger
fold, and they are nearly free: `msinf.model` builds the fitted likelihood as
`mix_over_nodes(sum_by_study(...))`, so the training log-density already
decomposes exactly into fourteen per-study terms and PSIS-LOO over them gives
leave-one-study-out across all fourteen from a *single* fit, plus a Pareto k̂ per
study naming the influential ones. Two caveats. Dropping one of fourteen studies
is a large perturbation, so expect some k̂ > 0.7, whose only honest remedy is a
refit. And PSIS leaves the held-out study's `log_eir` informed, so it estimates
the predictive *conditional on transmission* — a different question from the
study fold, and closer to the one `age_band` used to ask.

### The gates

A bad birthdates candidate raises or hangs. A bad candidate here *runs* and
returns plausible-looking nonsense. So before a single second is spent sampling,
the curves are evaluated at a few prior draws and a spread of EIRs and asked
whether they are finite, whether prevalence is a probability, whether incidence
is in a band a person could experience, and whether prevalence rises with
transmission. A candidate that fails is one cheap `Result` carrying every reason
it failed, not a wasted fit -- and it stays in the report, because the models
worth looking at are often the ones that failed.

### The fit

**One fit per split per candidate**, at roughly 20 minutes each: 4 chains, 400
warmup, 400 draws, f64. f64 is not optional -- a novel curve family at f32
through two sequential scan recursions produces NaNs, and the scoring pass
re-runs those recursions once per retained draw. `--fit-timeout` defaults to
20min × splits × 3.

Every number comes from `msinf.evaluate`, the same entry point msinf's own
`+experiment=holdout_*` runs go through. If a candidate's `elpd_holdout` were
not comparable with the number msinf reports for the incumbent, the search would
not be measuring what the report claims.

| metric | what it answers |
| --- | --- |
| `<split>.elpd_holdout`, `_incidence`, `_prevalence` | held-out density with the random effects marginalised exactly as the fitted likelihood marginalises them. Prevalence is scored per row; incidence per **cell**, since `v_jk` does not factorise. `_per_cell` and `_per_row` are the same numbers per unit. |
| `<split>.elpd_holdout_se` | that total's standard error over its units, so a total is not read as if it were exact. |
| `<split>.pointwise_holdout_*` | the per-unit vectors the totals sum, in the fold's own order — the only vector-valued metrics here. Two candidates under the same split line up elementwise, which is what lets the table's `vs best` column pair them. |
| `<split>.coverage90_holdout_*` | how often the held-out count fell inside the 90% predictive interval. |
| `<split>.kl.<parameter>`, `kl_min`, `unidentified` | the information the data added to each of the *candidate's own* parameters, in nats, unconstrained. The scaffolding's parameters are excluded: they are identical for every candidate and counting them would flatter a model that declared nothing. |
| `rhat_max`, `ess_tail_min`, `divergences` | the worst of the splits, not the average: a candidate that converged under one fold and not another has not converged. |
| `mcmc_seconds`, `total_seconds`, `parameters` | what it cost, and how much of it the candidate declared. |

Prior draws for the identifiability comparison are taken with the candidate's
`curves` swapped for a stub. Drawing them through the real thing would mean
thousands of full equilibrium solves at prior parameter values, most of them
numerically hostile and none of them wanted.

### Reading a run

```
  #     from  elpd/study elpd/age_b cover90  kl_min unid  rhat ess_tail   div   mins pars
  6  scratch      -412.7     -388.1   0.905    1.21    0  1.01      412     0   38.2    7
  9  griffin      -418.3     -391.4   0.917    0.44    0  1.02      288     3   61.7   19
  0*  scratch     -466.9     -452.0   0.882    2.31    0  1.01      501     0   31.4    3
  1*  griffin     -421.5     -397.6   0.911    0.31    1  1.03      201     8   58.9   18
  4  griffin      GateError: prevalence in the 2-10y window is flat in EIR: ...
```

Ranked on the age-band split; `*` marks the models the searches started from,
fitted alongside their proposals so the run says whether either search beat what
it began with, and `from` says which search found a row. Everything, model
source included, is in `runs/malaria/report.jsonl`.

### Not yet: the analysis bench

There is no `--analysis`. What an agent wants here is not birthdates' pandas
namespace but a simulator bench -- run the candidate, look at the emergent
curves, check them against the gates before proposing -- and that is deferred.
