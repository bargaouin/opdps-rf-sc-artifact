#!/usr/bin/env bash
#SBATCH --job-name=rfc-flow-smoke
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

source /path/to/opdps-rf-sc-v100/bin/activate
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

CFG="$REPO/configs/anchorflow_v100.yaml"
COARSE="$RUN_ROOT/proposed_v100/coarse/stage1_final_best.pt"
STATS="$RUN_ROOT/stage2_shared/residual_stats.json"

echo "===== GAUSSIAN RF SMOKE ====="

python scripts/train_anchor_flow.py \
  --config "$CFG" \
  --cache-dir "$CACHE_DIR" \
  --coarse-checkpoint "$COARSE" \
  --residual-stats "$STATS" \
  --run-dir "$RUN_ROOT/gaussian_rf_smoke" \
  --mode gaussian \
  --steps 200 \
  --device cuda

echo
echo "===== MIXED ANCHOR FLOW SMOKE ====="

python scripts/train_anchor_flow.py \
  --config "$CFG" \
  --cache-dir "$CACHE_DIR" \
  --coarse-checkpoint "$COARSE" \
  --residual-stats "$STATS" \
  --run-dir "$RUN_ROOT/anchorflow_smoke" \
  --mode mixed \
  --steps 200 \
  --device cuda

echo
echo "SMOKE TEST COMPLETE"
