# Find a malaria model that fits better than Griffin

Work in /home/athapar/code/discovery (branch `longer-runs`), with msinf at
/home/athapar/code/msinf. Read the files named below, then show me a plan before
writing code. Ask me if something here is wrong or blocks you. Treat this as my
ideas, not a spec: if another design reaches the goal sooner, propose it and say
why.

## Goal

Agentic proposals that fit the data better than the incumbent (Griffin,
`--target curves --start griffin`). My best so far, from
`python -m examples.malaria.evaluation`:

    Δelpd holdout +0.7 ± 0.4 | Δelpd train +0.6 ± 0.7 | 47.5 min | 15 pars | ESS_tail 80 | 1 unidentified

Both differences are inside 2 SE, so it isn't an improvement yet. **Done** means:
a candidate whose paired Δelpd_holdout vs Griffin is > 2 SE (`Z` in
evaluation.py), confirmed by a refit at 400 warmup / 400 samples, with r̂ ≤ 1.01,
no divergences, and no more unidentified parameters than Griffin.

Only the agentic arm matters now: `agentic.py` run through OpenCode and Codex.
Keep the i.i.d./AgentSearch path in `__main__` and the Claude arm working, but
don't spend effort on them.

## What exists (read first)

- `examples/malaria/agentic.py`: one isolated sandbox per agent, runs the CLI,
  collects `proposal.py` into `<out>/proposals/` plus transcripts, usage and
  failures. `opencode.py` and `codex.py` are thin wrappers around it.
- `examples/malaria/__main__.py`: `--proposals <dir>` fits a saved round and
  writes `report.jsonl`. `--no-search` fits only the start model.
- `fit.py` / `runner.py`: one subprocess per candidate. It runs the gates, then
  NUTS per split via `msinf.evaluate`, then writes metrics, pointwise elpd,
  PPC/calibration PNGs and `study.samples.npz`.
- `evaluation.py`, `frontier.py`, `sweep.py`: paired Δelpd vs the incumbent.
  `frontier.ARMS` is hard-coded to old remote run paths.
- `initial.py` (BRIEF, STARTS), `analysis.py` (the training rows, with the
  holdout removed), `gates.py`, `examples/estimators.py` (`elpd_diff`).
- msinf: `scripts/run_inference.py` (how I fit Griffin) and
  `src/msinf/evaluate.py`. `evaluate.py` already simulates posterior-predictive
  replicates per row and has per-draw log-lik in `_per_draw_terms`.
  `src/msinf/plotting/trajectories.py` (`plot_all_study_trajectories`,
  `_fitted_draws`) works for Griffin only, because it solves dmeq with
  Griffin's parameters.

Inputs:
- Table: `/home/athapar/code/msinf/artifacts/inputs/study_data_2014.csv`
  (223 rows, already ≤ 20 y). `data.DEFAULT_TABLE` points at a file that isn't
  here, so set `MALARIA_TABLE`.
- EIR priors: `/home/athapar/code/msinf/artifacts/inputs/eir_site_priors_griffin2014.csv`
- Griffin posterior:
  `/home/athapar/code/msinf/outputs/2026-10-14/15-54-43/numpyro_jamie_samples.npz`
  (config beside it). It is msinf's **full-data** fit (400/400, seed 42), so it
  saw the held-out studies. Use it to cross-check the pipeline's own Griffin fit
  and for plots I look at. Never put it, or anything derived from it, in a
  sandbox.

## Fix first

1. `runner.py:306` reads `result.fit_result`, but `msinf.evaluate.EvaluationResult`
   has no such field, so a completed fit crashes while writing `samples.npz`. Fix
   it, then prove one fit runs end to end.
2. The MCMC budget is defined in `__main__` (150/150), `agentic.py` (tells agents
   400/400) and `fit.Budget` (400/400). Make one source of truth.
   150 warmup + 150 samples × 4 chains is the screening budget.
3. `FIT_MINUTES = 20` sets the fit timeout (×3) and is what agents are told, but
   my best candidate took 47.5 min. Measure Griffin at the screening budget and
   set it from that.
4. `codex.py` defaults to `gpt-5`. Use luna: `gpt-6-luna` (also listed:
   `gpt-5.6-luna`).

## The design change

The agentic arm was built to *compare* proposers fairly, so agents can't fit,
can't run the gates and see no results. I now just want better models, so relax
all of that, with one exception that stays absolute:

**Agents never see the held-out fold.** That means not its rows, not any
candidate's holdout scores, and not posteriors or plots from fits that included
it. We select on 35 holdout units. If agents tune against them round after
round, Δelpd_holdout stops meaning anything. Holdout numbers go only to reports
for me.

Keep the rest of the isolation: sandboxes outside the repo, no web, no
subagents, no access to the repo or old runs. Agents learn about earlier rounds
only through the feedback pack below.

## Work, in order

### 1. Baseline
- Fit Griffin on the training fold at 150/150 (`--no-search --start griffin`).
  Its `study.samples.npz` is the incumbent posterior agents may see.
- Compare its parameter medians with the msinf full-data posterior and flag big
  gaps.
- Record elpd_train and elpd_holdout ± se, runtime, ESS_tail and r̂.

### 2. Feedback pack (the main lever)
Add `examples/malaria/feedback.py`. It reads finished rounds (any directory with
a `report.jsonl`) and writes a `feedback/` directory. `agentic.py --history
<round dirs or a root>` copies that directory into every sandbox, and
BRIEF/ROLE describe it. Text and CSV come first, because GLM-flash is text-only.
PNGs are an extra for Codex. Everything in it is training-fold only:

- **Incumbent:** per-parameter posterior summary (median, 90% CI, KL vs prior),
  its fitted table (below), and `plot_all_study_trajectories` PNGs from the
  training-fold samples.
- **Fitted incidence and prevalence per training row** for every fitted
  candidate, from the candidate's *own* curves:
  - Columns: observed rate, posterior-predictive median and 90% interval, and
    pointwise elpd (incidence per cell, as msinf scores it).
  - A residual summary by age band, EIR category, stream and study.
  - msinf.evaluate already simulates these replicates. Have it return a per-row
    summary instead of solving again, and have the runner save that as an
    artifact.
- **Leaderboard:** for each earlier candidate:
  - its diff vs Griffin;
  - its paired Δ vs Griffin on the training units ± se (`estimators.elpd_diff`);
  - parameter count, unidentified parameters, r̂, ESS_tail, divergences, runtime.

  Prefer PSIS-LOO on the training units (the per-draw log-lik already exists)
  over raw elpd_train, so extra parameters aren't rewarded. Fall back to
  elpd_train if LOO isn't cheap.
- **Failures:** gate reasons and fit errors, so mistakes aren't repeated.
- **Size:** keep it bounded (top-k plus failures) so it fits comfortably in a
  196 608-token context.

Also make a report for me: a command that takes any round directories and
prints the table I quoted above (paired Δelpd holdout and train vs Griffin,
min, pars, ESS_tail, unid), reusing `evaluation.differences`/`table`. Today that
only works for the hard-coded `frontier.ARMS`.

### 3. Propose here, fit anywhere, feed back later
A round is a directory. It holds `proposals/` and `round.json` after proposing,
and `report.jsonl` plus artifacts after fitting. Proposing and fitting never
have to happen on the same machine or at the same time. Each new round's
`--history` uses whatever rounds have finished.

- **Local fit:** the existing
  `python -m examples.malaria --proposals <round>/proposals --out <round>`.
- **Slurm fit:**
  - Setup: the head node is `wpia-lxhn.hpc.dide.ic.ac.uk`. Its NFS is visible to
    every compute node (96 CPUs each) but not to this machine.
  - Add `examples/malaria/slurm/` with an sbatch script that runs the same fit
    command with `--concurrency` = `$SLURM_CPUS_PER_TASK / chains`.
  - Add a small script that rsyncs a round up, submits it over ssh, and later
    rsyncs `report.jsonl` and artifacts back.
  - Host and remote path come from env vars with defaults.
  - Check memory per fit (`sacct -o MaxRSS`) on the first job before raising
    concurrency.
  - Shell only, no job-queue framework.
- **Two stages:** screen every candidate at 150/150. Refit only the top few
  (by training score) at 400/400, ideally on a second holdout seed too, before
  calling anything a win.

### 4. Let the agents explore
- **OpenCode.** The model is GLM-flash on llama.cpp on a Slurm node (already
  running; don't manage it). It is reached at `opencode.json`'s baseURL, with 4
  slots × 196 608 context.
  - Build each agent's config from the repo's `opencode.json` (model, limits,
    32 768 output, reasoning variants) instead of the copy hard-coded in
    `opencode_config`.
  - Use the `high` or `max` variant (check `opencode run --help` for how to
    select it), with `--concurrency 4` to match the slots.
  - Don't cap tool calls or thinking. Keep a wall-clock cap only to stop hung
    sessions: make it hours (e.g. 4 h), tell the agent the real number, and keep
    the "write proposal.py early, then improve it" rule.
- **Codex.** `gpt-6-luna` with high reasoning effort via `-c
  model_reasoning_effort=...` (check that it applies under
  `--ignore-user-config`), and the same generous timeout.
- **Gate check.** Give every sandbox `./check proposal.py`, which splices the
  proposal onto the parent and runs `gates.check` exactly as the runner does.
  Install jax and numpyro into the sandbox venv, but not msinf, which would
  point at the repo and its data. Update ROLE (which says numpyro is missing).
- **Optional, behind a flag, only after 1–3 work:** a quick training-only fit in
  the sandbox (Pathfinder or short NUTS on `data/train.csv`), so an agent can
  compare its candidate to Griffin's training score. The sandbox must still
  contain no held-out row.

### 5. README
Add a "Searching for a better model" section to `examples/README.md` with
copy-pasteable commands for:
- (a) OpenCode proposals + local fit
- (b) OpenCode proposals + Slurm fit
- (c) Codex/luna proposals
- (d) a multi-round loop where round k+1 uses feedback from finished rounds
  while later fits are still running
- (e) the comparison table vs Griffin

List the env vars (`MALARIA_TABLE`, `MALARIA_EIR_PRIORS`, `LLM_API_KEY`,
cluster host/path) and a check that the GLM server is reachable (`curl
$URL/models`, and `/props` showing 4 slots).

## How to write the code
- Small and readable. Match the surrounding style (docstrings that say why,
  argparse groups). Reuse `__main__.load/save`, `agentic.collect`,
  `report.read/write`, `estimators.elpd_diff`, `evaluation.differences` and
  `msinf.evaluate` rather than writing parallel versions.
- Don't break existing flags or tests. Add tests for the feedback pack (above
  all: no held-out row or holdout metric ever reaches a sandbox) and for the
  Slurm command construction.
- Keep msinf changes minimal and backward compatible, and list them.
- Run `uv run pytest` and one real smoke run (2 agents, short fit) before saying
  it works. If you can't run something from here (e.g. Slurm), say so and give
  me the exact command.

## When you finish, report
Files changed, baseline Griffin numbers, smoke-test output, and anything you did
differently from this brief and why.
