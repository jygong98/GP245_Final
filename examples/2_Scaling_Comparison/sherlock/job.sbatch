#!/bin/bash
# Parameterized SLURM job for one Python FDFK scaling case.
#
# Submit via submit_all.sh, which sets resource limits and exports:
#   LABEL, SUITE, THREADS, NX, NZ, NT, SPATIAL_FACTOR (optional), FDFK_PYTHON
#
# Manual example:
#   sbatch --job-name=python_t8 --cpus-per-task=8 --time=04:00:00 --mem=32G \
#     --export=ALL,LABEL=python_t8,SUITE=thread,THREADS=8,NX=1001,NZ=251,NT=4001 \
#     job.sbatch

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=serc
# SBATCH --mail-type=END,FAIL
# SBATCH --mail-user=jygong@stanford.edu

set -euo pipefail

SHERLOCK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${SHERLOCK_DIR}/logs" "${SHERLOCK_DIR}/results"

# --- Required case variables (set by submit_all.sh or sbatch --export) ---
: "${LABEL:?LABEL must be set (e.g. python_t8 or python_s4)}"
: "${SUITE:?SUITE must be set (thread or spatial)}"
: "${THREADS:?THREADS must be set}"
: "${NX:?NX must be set}"
: "${NZ:?NZ must be set}"
: "${NT:=4001}"

export FDFK_PYTHON="${FDFK_PYTHON:-/oak/stanford/groups/sklemp/jygong/GP245/FDFK_Python}"
EXAMPLE_DIR="${FDFK_PYTHON}/examples/2_Scaling_Comparison"
RUNNER="${EXAMPLE_DIR}/run_python_scaling.py"
RESULT_JSON="${SHERLOCK_DIR}/results/${LABEL}.json"

# --- Thread environment (must be set before Python/Numba starts) ---
export NUMBA_NUM_THREADS="${THREADS}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Python / conda (adjust for your Sherlock setup) ---
# Option A: module + conda activate
# module load system
# module load anaconda
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate GP245_Final
# PYTHON="$(which python)"

#source /home/users/jygong/miniconda3/etc/profile.d/conda.sh
#conda init
#conda activate fdfk

PYTHON="/home/users/jygong/miniconda3/envs/fdfk/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python not found at ${PYTHON}" >&2
  echo "Set PYTHON or CONDA_ENV_ROOT, or uncomment the module/conda block above." >&2
  exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: Runner not found at ${RUNNER}" >&2
  echo "Set FDFK_PYTHON to your synced project root on Sherlock/Oak." >&2
  exit 1
fi

cd "${EXAMPLE_DIR}"

echo "=== ${LABEL} (${SUITE}) ==="
echo "Host: $(hostname)"
echo "Grid: ${NX} x ${NZ}, nt=${NT}, threads=${THREADS}"
echo "Python: ${PYTHON}"
echo "Output: ${RESULT_JSON}"
echo "Started: $(date -Is)"

ARGS=(
  --worker
  --nx "${NX}"
  --nz "${NZ}"
  --nt "${NT}"
  --thread-count "${THREADS}"
  --label "${LABEL}"
  --worker-suite "${SUITE}"
  --output-json "${RESULT_JSON}"
)

if [[ -n "${SPATIAL_FACTOR:-}" ]]; then
  ARGS+=(--spatial-factor "${SPATIAL_FACTOR}")
fi

"${PYTHON}" "${RUNNER}" "${ARGS[@]}"

echo "Finished: $(date -Is)"
echo "Wrote ${RESULT_JSON}"
