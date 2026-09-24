#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RFC_VENV="${RFC_VENV:-$HOME/venvs/rfchallenge-2021}"
RF_PYTHON_BIN="${RF_PYTHON_BIN:-python3}"

"$RF_PYTHON_BIN" -m venv "$RFC_VENV"
source "$RFC_VENV/bin/activate"
python -m pip install --upgrade 'pip<25' wheel setuptools
pip install -r "$REPO/requirements-rfchallenge.txt"

echo "Created RFChallenge decoding environment: $RFC_VENV"
python - <<'PY'
import numpy
import sigmf
import commpy
print('numpy', numpy.__version__)
print('sigmf import: OK')
print('commpy import: OK')
PY
