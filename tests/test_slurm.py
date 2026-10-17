"""The cluster scripts, as the commands they would run.

Nothing here reaches a cluster: `DRY_RUN=1` prints what `remote.sh` would do and
`PYTHON=echo` makes the batch script print the fit it would start. What is worth
pinning is that a round lands at the same relative path on both sides, that the
fit on a node is the local fit with the node's CPUs in `--concurrency`, and that
flags reach `python -m examples.malaria` untouched.
"""

import os
import subprocess
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[1] / "examples" / "malaria" / "slurm"


def remote(*args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SLURM / "remote.sh"), *args],
        capture_output=True, text=True, env=os.environ | {"DRY_RUN": "1"} | env, check=False,
    )


def test_submit_copies_the_proposals_up_and_sbatches_the_round_in_place():
    done = remote("submit", "runs/beat/r01/", "--warmup", "400", "--samples", "400",
                  MALARIA_HOST="head", MALARIA_REMOTE="work/discovery", MALARIA_SBATCH="--partition=big")
    assert done.returncode == 0, done.stderr
    push, submit = done.stdout.strip().splitlines()
    assert push.startswith("rsync ") and "runs/beat/r01/proposals/ head:work/discovery/runs/beat/r01/proposals/" in push
    assert submit.startswith("ssh head ")
    assert "sbatch\\ --output=runs/beat/r01/slurm-%j.out\\ --partition=big" in submit
    assert "fit.sbatch\\ runs/beat/r01\\ --warmup\\ 400\\ --samples\\ 400" in submit


def test_fetch_brings_back_everything_but_the_proposals():
    done = remote("fetch", "runs/beat/r01", MALARIA_HOST="head", MALARIA_REMOTE="work/discovery")
    assert done.returncode == 0, done.stderr
    assert "--exclude proposals head:work/discovery/runs/beat/r01/ " in done.stdout


@pytest.mark.parametrize("round_", ["/abs/round", ""])
def test_a_round_must_be_relative_to_the_repository(round_):
    done = remote("submit", round_)
    assert done.returncode == 2
    assert "relative to the repository root" in done.stderr


def test_code_never_deletes_on_the_cluster():
    done = remote("code", MALARIA_HOST="head")
    assert done.returncode == 0, done.stderr
    assert "--delete" not in done.stdout
    assert "--exclude .venv" in done.stdout and "uv\\ sync" in done.stdout


def test_the_batch_job_is_the_local_fit_with_the_nodes_cpus():
    done = subprocess.run(
        ["bash", str(SLURM / "fit.sbatch"), "runs/beat/r01", "--warmup", "400"],
        capture_output=True, text=True, check=False,
        env=os.environ | {"PYTHON": "echo", "SLURM_CPUS_PER_TASK": "96", "SLURM_SUBMIT_DIR": str(SLURM)},
    )
    assert done.returncode == 0, done.stderr
    fit = done.stdout.strip().splitlines()[-1]
    assert fit == (
        "-m examples.malaria --proposals runs/beat/r01/proposals --out runs/beat/r01 "
        "--chains 4 --concurrency 24 --warmup 400"
    )
