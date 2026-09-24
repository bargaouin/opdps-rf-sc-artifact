#!/usr/bin/env bash
#SBATCH --job-name=rfc-wave84-eval
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /path/to/opdps_rf_sc
source slurm/env.sh

source /path/to/opdps-rf-sc-v100/bin/activate
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

CKPT="$RUN_ROOT/baseline_waveunet_parammatch_v100/coarse/best.pt"
CFG="$RUN_ROOT/baseline_waveunet_parammatch_v100/coarse/eval_config_from_checkpoint.yaml"
OUT="$RUN_ROOT/evals/baseline_waveunet_parammatch_v100"

mkdir -p "$OUT"

echo "===== CHECKPOINT ====="
ls -lh "$CKPT"

echo
echo "===== EXTRACT CONFIG FROM CHECKPOINT ====="

python - "$CKPT" "$CFG" <<'PY'
import sys
import torch
import yaml

ckpt_path, cfg_path = sys.argv[1], sys.argv[2]

ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False,
)

cfg = ckpt["config"]

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("checkpoint:", ckpt_path)
print("step:", ckpt.get("step"))
print("model kind:", cfg["model"]["kind"])
print("config saved:", cfg_path)
PY

echo
echo "===== OFFICIAL 2200-EXAMPLE EVALUATION ====="

python scripts/eval_sep_val.py \
    --cache-dir "$CACHE_DIR" \
    --config "$CFG" \
    --coarse-checkpoint "$CKPT" \
    --output-dir "$OUT" \
    --batch-size 4 \
    --device cuda

echo
echo "===== SUMMARY ====="
cat "$OUT/summary.json"

echo
echo "===== PER-SINR ====="
cat "$OUT/summary_by_sinr.csv"
