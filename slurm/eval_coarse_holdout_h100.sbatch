#!/usr/bin/env bash
#SBATCH --job-name=rfc-coarse-val
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /path/to/opdps_rf_sc
source slurm/env.sh

source /path/to/opdps-rf-sc/bin/activate
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

python scripts/eval_coarse_holdout_matched.py \
  --config "$REPO/configs/anchorflow_v100.yaml" \
  --cache-dir "$CACHE_DIR" \
  --coarse-checkpoint "$RUN_ROOT/proposed_v100/coarse/stage1_final_best.pt" \
  --batch-size 4 \
  --batches 4 \
  --device cuda
