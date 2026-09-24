#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/venvs/opdps-rf-sc}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
# PyTorch wheels are cluster/CUDA dependent. If torch is already supplied by a module,
# install this repo with --no-deps after activating that environment instead.
pip install -e "${REPO}[test]"

echo "Created $VENV"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
