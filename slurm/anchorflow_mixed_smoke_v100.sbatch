#!/usr/bin/env bash
#SBATCH --job-name=rfc-anchor-smoke
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

python scripts/train_anchor_flow.py \
  --config "$REPO/configs/anchorflow_v100.yaml" \
  --cache-dir "$CACHE_DIR" \
  --coarse-checkpoint "$RUN_ROOT/proposed_v100/coarse/stage1_final_best.pt" \
  --residual-stats "$RUN_ROOT/stage2_shared/residual_stats.json" \
  --run-dir "$RUN_ROOT/anchorflow_mixed_smoke" \
  --mode mixed \
  --steps 200 \
  --device cuda
