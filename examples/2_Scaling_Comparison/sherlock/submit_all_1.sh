#!/bin/bash
# Submit all Python FDFK scaling benchmarks as individual SLURM jobs.
#
# Usage (from this directory on Sherlock):
#   bash submit_all.sh
#
# Optional overrides:
#   FDFK_PYTHON=/path/to/FDFK_Python bash submit_all.sh
#   SLURM_ACCOUNT=mysponsor bash submit_all.sh
#   SLURM_PARTITION=owners bash submit_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/results"

FDFK_PYTHON="${FDFK_PYTHON:-/oak/stanford/groups/sklemp/jygong/GP245/FDFK_Python}"
NT=4001

# SLURM options shared by all jobs (override via environment).
SLURM_PARTITION="${SLURM_PARTITION:-serc}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-sklemp}"
SBATCH_ACCOUNT=()
if [[ -n "${SLURM_ACCOUNT}" ]]; then
  SBATCH_ACCOUNT=(--account="${SLURM_ACCOUNT}")
fi

JOB_SCRIPT="${SCRIPT_DIR}/job.sbatch"
if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: ${JOB_SCRIPT} not found" >&2
  exit 1
fi

submit_job() {
  local job_name="$1"
  local cpus="$2"
  local time_limit="$3"
  local mem="$4"
  local export_vars="$5"

  sbatch \
    --job-name="${job_name}" \
    --partition="${SLURM_PARTITION}" \
    "${SBATCH_ACCOUNT[@]}" \
    --cpus-per-task="${cpus}" \
    --time="${time_limit}" \
    --mem="${mem}" \
    --output="${SCRIPT_DIR}/logs/${job_name}-%j.out" \
    --error="${SCRIPT_DIR}/logs/${job_name}-%j.err" \
    --export=ALL,FDFK_PYTHON="${FDFK_PYTHON}",NT="${NT}",${export_vars} \
    "${JOB_SCRIPT}"
}

echo "Submitting thread scaling jobs (python_t1 .. python_t12) ..."
for t in 1 2 4 8 16 32; do
    mem="$((t * 8))G"
    submit_job \
    "python_t${t}" \
    "${t}" \
    "01:30:00" \
    "${mem}" \
    "LABEL=python_t${t},SUITE=thread,THREADS=${t},NX=1001,NZ=251"
done


echo "Done. Monitor with: squeue -u \$USER"
echo "After all jobs finish: bash merge_results.sh"
