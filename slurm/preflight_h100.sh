#!/usr/bin/env bash
#SBATCH --job-name=rfc-preflight
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load python/3.11.6

REPO=/path/to/opdps_rf_sc
VENV=/path/to/opdps-rf-sc

mkdir -p "$REPO/logs"

source "$VENV/bin/activate"
cd "$REPO"

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

echo "===== SYSTEM ====="
hostname
which python
python --version
nvidia-smi

echo
echo "===== PYTORCH ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

echo
echo "===== COARSE MODEL PREFLIGHT ====="
python scripts/h100_preflight.py \
    --config configs/proposed.yaml \
    --stage coarse

echo
echo "===== DIFFUSION MODEL PREFLIGHT ====="
python scripts/h100_preflight.py \
    --config configs/proposed.yaml \
    --stage diffusion

echo
echo "PREFLIGHT COMPLETE"
