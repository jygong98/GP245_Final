#!/bin/bash
#SBATCH --job-name=fdfk_bench_test1
#SBATCH --output=output/benchmark/slurm-%j.out
#SBATCH --error=output/benchmark/slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=16GB

# FDFK2D vs FDFK_Python hybrid benchmark — examples/test1 (1-layer, 200 km profile)
#
# Submit from this directory:
#   sbatch submit.sbatch
#
# Override paths if needed:
#   sbatch --export=ALL,FDFK_ROOT=/path/to/FDFK2D,FDFK_PYTHON=/path/to/FDFK_Python submit.sbatch

set -euo pipefail
BENCH_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BENCH_DIR"

module load ifort
module load imkl
module load fftw

export FDFK_ROOT="${FDFK_ROOT:-/oak/stanford/groups/sklemp/jygong/GP245/FDFK2D/FDFK2D}"
export FDFK_PYTHON="${FDFK_PYTHON:-/oak/stanford/groups/sklemp/jygong/GP245/FDFK_Python}"
export CONDA_ENV="${CONDA_ENV:-mantleeq}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

bash run_job.sh
