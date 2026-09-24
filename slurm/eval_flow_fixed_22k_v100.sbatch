#!/usr/bin/env bash
#SBATCH --job-name=rfc-fixed22
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

python scripts/eval_flow_fixed_holdout.py \
  --config "$REPO/configs/anchorflow_v100.yaml" \
  --cache-dir "$CACHE_DIR" \
  --coarse-checkpoint "$RUN_ROOT/proposed_v100/coarse/stage1_final_best.pt" \
  --residual-stats "$RUN_ROOT/stage2_shared/residual_stats.json" \
  --gaussian "$RUN_ROOT/gaussian_rf_v100/snapshot_22k.pt" \
  --anchor "$RUN_ROOT/anchorflow_v100/snapshot_22000.pt" \
  --examples 256 \
  --batch-size 4 \
  --seed 20260921 \
  --solver heun \
  --ode-steps 4 \
  --source-weight 1.0 \
  --device cuda
