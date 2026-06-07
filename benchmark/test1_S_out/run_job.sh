#!/usr/bin/env bash
# Run Fortran + Python hybrid benchmarks and write timing/speedup summary.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH_DIR"

# --- HPC paths (override via environment before sbatch) ---
FDFK_ROOT="${FDFK_ROOT:-/oak/stanford/groups/sklemp/jygong/GP245/FDFK2D/FDFK2D}"
FDFK_PYTHON="${FDFK_PYTHON:-/oak/stanford/groups/sklemp/jygong/GP245/FDFK_Python}"
CONDA_ENV="${CONDA_ENV:-mantleeq}"

mkdir -p output/fortran output/python output/benchmark snapshots

echo "============================================================"
echo " FDFK2D benchmark case: $(basename "$BENCH_DIR")"
echo " Host: $(hostname)  Date: $(date -Iseconds)"
echo " FDFK_ROOT=$FDFK_ROOT"
echo " FDFK_PYTHON=$FDFK_PYTHON"
echo " OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
echo "============================================================"

# --- Resolve FDFK2D executable ---
if command -v FDFK2D >/dev/null 2>&1; then
  FDFK2D_BIN="$(command -v FDFK2D)"
elif [[ -x "${FDFK_ROOT}/bin/FDFK2D" ]]; then
  FDFK2D_BIN="${FDFK_ROOT}/bin/FDFK2D"
elif [[ -x "${HOME}/bin/FDFK2D" ]]; then
  FDFK2D_BIN="${HOME}/bin/FDFK2D"
else
  echo "ERROR: FDFK2D executable not found. Set FDFK_ROOT or add ~/bin/FDFK2D to PATH." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${FDFK_ROOT}/lib_intel:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

# --- Fortran (pipe Y for optional 2-D geometry warning prompt) ---
echo ""
echo ">>> Fortran FDFK2D: ${FDFK2D_BIN}"
printf 'Y\n' | /usr/bin/time -f "WALL_SEC=%e MAXRSS_KB=%M" \
  -o output/benchmark/fortran.time \
  "${FDFK2D_BIN}" ./input inpar.dat ./output/fortran seis \
  2>&1 | tee output/benchmark/fortran.log

# Copy stripped models for Python if Fortran just created them
for f in input/*_without_pml.su; do
  [[ -e "$f" ]] || continue
  :
done

# --- Python hybrid ---
echo ""
echo ">>> Python hybrid (FDFK_Python)"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${FDFK_PYTHON}:${PYTHONPATH:-}"
python3 run_python_benchmark.py 2>&1 | tee output/benchmark/python.log

# --- Summary ---
python3 summarize_benchmark.py 2>&1 | tee output/benchmark/benchmark_summary.log

echo ""
echo "Done. Results in output/benchmark/"
