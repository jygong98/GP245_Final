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

echo "Submitting spatial scaling jobs (python_s1 .. python_s16) ..."

declare -A SPATIAL_NX=(
  [1]=1001 [2]=1001 [4]=2001 [8]=2001 [16]=4001
)
declare -A SPATIAL_NZ=(
  [1]=251 [2]=501 [4]=501 [8]=1001 [16]=1001
)
declare -A SPATIAL_TIME=(
  [1]=01:00:00 [2]=01:00:00 [4]=01:30:00 [8]=02:00:00 [16]=02:30:00
)
declare -A SPATIAL_MEM=(
  [1]=32G [2]=32G [4]=48G [8]=64G [16]=128G
)

for factor in 1 2 4 8 16; do
  submit_job \
    "python_s${factor}" \
    8 \
    "${SPATIAL_TIME[$factor]}" \
    "${SPATIAL_MEM[$factor]}" \
    "LABEL=python_s${factor},SUITE=spatial,THREADS=8,NX=${SPATIAL_NX[$factor]},NZ=${SPATIAL_NZ[$factor]},SPATIAL_FACTOR=${factor}"
done

echo "Done. Monitor with: squeue -u \$USER"
echo "After all jobs finish: bash merge_results.sh"
