#!/usr/bin/env bash
#SBATCH --job-name=rfc-c64
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /path/to/opdps_rf_sc

source slurm/env.sh

V100_VENV=/path/to/opdps-rf-sc-v100
source "$V100_VENV/bin/activate"

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

CONFIG="$REPO/configs/proposed_v100.yaml"
COARSE_CKPT="$RUN_ROOT/proposed_v100/coarse/stage1_final_best.pt"
OUT="$RUN_ROOT/evals/debug_coarse_same64"

mkdir -p "$OUT" "$REPO/logs"

echo "===== DEVICE ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
PY

echo
echo "===== COARSE SAME-SUBSET EVALUATION ====="

python scripts/eval_sep_val.py \
    --cache-dir "$CACHE_DIR" \
    --config "$CONFIG" \
    --coarse-checkpoint "$COARSE_CKPT" \
    --output-dir "$OUT" \
    --batch-size 4 \
    --max-examples 32 \
    --device cuda

echo
echo "===== SUMMARY ====="
cat "$OUT/summary.json"

echo
echo "===== PER-SINR ====="
cat "$OUT/summary_by_sinr.csv"
