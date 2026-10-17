#!/usr/bin/env bash
# Fit rounds on the Slurm cluster from this machine: sync, submit, fetch.
#
#   examples/malaria/slurm/remote.sh code                     # copy this checkout and msinf up, uv sync there
#   examples/malaria/slurm/remote.sh submit <round> [flags]   # copy the round's proposals up, sbatch its fit
#   examples/malaria/slurm/remote.sh status                   # your jobs
#   examples/malaria/slurm/remote.sh fetch <round>            # bring its report, fit.json and artifacts back
#
# The head node shares a filesystem with every compute node but not with this
# machine, so a round crosses twice: up as proposals, back as a report. `flags`
# go to `python -m examples.malaria`, e.g. `--warmup 400 --samples 400`.
# A <round> is a path relative to the repository root, the same on both sides.
#
# MALARIA_HOST     SSH target (user@host)                [athapar@wpia-lxhn.hpc.dide.ic.ac.uk]
# MALARIA_REMOTE   the checkout there, relative to ~     [code/discovery]
# MALARIA_SBATCH   extra sbatch flags                    e.g. "--partition=x --mem=300G"
# DRY_RUN=1        print the commands instead of running them
set -euo pipefail

host=${MALARIA_HOST:-athapar@wpia-lxhn.hpc.dide.ic.ac.uk}
remote=${MALARIA_REMOTE:-code/discovery}
here=$(cd "$(dirname "$0")/../../.." && pwd)

run() {
    if [[ -n "${DRY_RUN:-}" ]]; then printf '%q ' "$@"; echo; else "$@"; fi
}

round() {
    [[ -n "${1:-}" && "$1" != /* ]] || { echo "a round is a path relative to the repository root, got '${1:-}'" >&2; exit 2; }
    echo "${1%/}"
}

case "${1:-}" in
    code)
        # Adds and updates; never deletes, so nothing that exists only there is lost.
        exclude=(--exclude .venv --exclude .git --exclude __pycache__ --exclude 'runs*/')
        run rsync -az "${exclude[@]}" "$here/" "$host:$remote/"
        run rsync -az "${exclude[@]}" --exclude outputs "$here/../msinf/" "$host:$remote/../msinf/"
        # Non-interactive SSH shells may skip the PATH setup used at login.
        run ssh "$host" "export PATH=\"\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH\"; cd $remote && uv sync --group examples --group malaria"
        ;;
    submit)
        path=$(round "${2:-}")
        shift 2
        run rsync -az --rsync-path="mkdir -p $remote/$path/proposals && rsync" \
            "$here/$path/proposals/" "$host:$remote/$path/proposals/"
        run ssh "$host" "cd $remote && sbatch --output=$path/slurm-%j.out ${MALARIA_SBATCH:-} examples/malaria/slurm/fit.sbatch $path $*"
        ;;
    status)
        run ssh "$host" "squeue --me"
        ;;
    fetch)
        path=$(round "${2:-}")
        run rsync -az --exclude proposals "$host:$remote/$path/" "$here/$path/"
        ;;
    *)
        sed -n '2,19p' "$0"
        exit 2
        ;;
esac
