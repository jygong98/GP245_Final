#!/usr/bin/env bash
# Create or update the project conda environment, then run a quick smoke test.
#
# Usage (from repo root):
#   bash scripts/setup_env.sh
#
# Requires: conda with conda-forge channel access.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="fdfk"
ENV_FILE="environment-minimal.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found."
  echo "Install Miniconda/Anaconda, or use pip in a Python 3.10–3.12 venv:"
  echo "  python -m venv .venv && source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing environment: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "Creating environment: $ENV_NAME"
  conda env create -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"

echo ""
echo "Verifying imports..."
python - <<'PY'
import numpy, scipy, matplotlib, numba, yaml, pytest
import seispy
from seispy.decon import deconit
print("OK:", f"numpy {numpy.__version__}, scipy {scipy.__version__}, seispy via {seispy.__file__}")
PY

echo ""
echo "Running core tests..."
python -m pytest tests/test_propagator_haskell.py tests/test_io_yaml.py -q

echo ""
echo "Environment ready. Activate with:"
echo "  conda activate $ENV_NAME"
