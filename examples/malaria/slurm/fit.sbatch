#!/bin/bash
# Fit one round of proposals on one Slurm node.
#
#   sbatch examples/malaria/slurm/fit.sbatch <round> [flags for python -m examples.malaria]
#
# Run from the checkout on the head node -- `remote.sh submit` does this from
# your own machine. It is the same command as a local fit, with `--concurrency`
# filled from the node: one fit is `--chains` chains of one thread each, so a
# 96-CPU node runs 24 at once. Check one job's memory before trusting that:
#
#   sacct -j <jobid> -o JobID,MaxRSS,Elapsed
#
# CHAINS       chains per fit                         [4]
# CONCURRENCY  fits at once                           [cpus / chains]
# PYTHON       the interpreter                        [.venv/bin/python]
#
#SBATCH --job-name=malaria-fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
set -euo pipefail

round=${1:?usage: sbatch fit.sbatch <round> [flags for python -m examples.malaria]}
shift
cd "${SLURM_SUBMIT_DIR:-.}"

chains=${CHAINS:-8}
cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}
concurrency=${CONCURRENCY:-$(( cpus / chains > 0 ? cpus / chains : 1 ))}

echo "fitting $round on $(hostname): $concurrency at once, $chains chains each, $cpus cpus"
"${PYTHON:-.venv/bin/python}" -m examples.malaria \
    --proposals "$round/proposals" --out "$round" \
    --chains "$chains" --concurrency "$concurrency" "$@"
