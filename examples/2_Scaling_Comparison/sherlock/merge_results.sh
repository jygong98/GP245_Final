#!/bin/bash
# Merge per-job result JSON files into python_scaling_times.json.
#
# Usage (after all SLURM jobs finish):
#   bash merge_results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
OUTPUT_JSON="${SCRIPT_DIR}/../python_scaling_times.json"

FDFK_PYTHON="${FDFK_PYTHON:-/oak/stanford/groups/sklemp/jygong/GP245/FDFK_Python}"
CONDA_ENV_ROOT="${CONDA_ENV_ROOT:-/home/users/${USER}/.conda/envs}"
PYTHON="${PYTHON:-${CONDA_ENV_ROOT}/GP245_Final/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  echo "Set PYTHON or activate GP245_Final before running merge." >&2
  exit 1
fi

if [[ ! -d "${RESULTS_DIR}" ]] || [[ -z "$(ls -A "${RESULTS_DIR}"/*.json 2>/dev/null || true)" ]]; then
  echo "ERROR: No result files in ${RESULTS_DIR}" >&2
  echo "Wait for SLURM jobs to finish, or check logs/ for failures." >&2
  exit 1
fi

"${PYTHON}" "${SCRIPT_DIR}/merge_results.py" \
  --results-dir "${RESULTS_DIR}" \
  --output "${OUTPUT_JSON}"

echo "Merged cache written to ${OUTPUT_JSON}"
