#!/usr/bin/env bash
#SBATCH --job-name=rfc-sampler-test
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source /path/to/opdps-rf-sc-v100/bin/activate

REPO=/path/to/opdps_rf_sc
RFBASE=/path/to/rfchallenge_singlechannel_starter
RUN_ROOT="$RFBASE/runs/opdps_rf_sc"
CACHE_DIR="$RFBASE/rfchallenge_cache_sc"

cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

CONFIG="$REPO/configs/proposed_v100.yaml"
COARSE="$RUN_ROOT/proposed_v100/coarse/best.pt"
DIFF="$RUN_ROOT/proposed_v100/operator_diffusion/best.pt"
STATS="$RUN_ROOT/stage2_shared/residual_stats.json"

echo "===== DPM-SOLVER++ CLIPPED ====="

python scripts/eval_sep_val.py \
  --cache-dir "$CACHE_DIR" \
  --config "$CONFIG" \
  --coarse-checkpoint "$COARSE" \
  --diffusion-checkpoint "$DIFF" \
  --residual-stats "$STATS" \
  --output-dir "$RUN_ROOT/evals/debug_dpm_clipped" \
  --sampler dpm \
  --solver-steps 20 \
  --posterior-samples 1 \
  --batch-size 4 \
  --max-examples 32 \
  --device cuda

echo
cat "$RUN_ROOT/evals/debug_dpm_clipped/summary.json"

echo
echo "===== DDIM CURRENT IMPLEMENTATION ====="

python scripts/eval_sep_val.py \
  --cache-dir "$CACHE_DIR" \
  --config "$CONFIG" \
  --coarse-checkpoint "$COARSE" \
  --diffusion-checkpoint "$DIFF" \
  --residual-stats "$STATS" \
  --output-dir "$RUN_ROOT/evals/debug_ddim_current" \
  --sampler ddim \
  --solver-steps 30 \
  --posterior-samples 1 \
  --batch-size 4 \
  --max-examples 32 \
  --device cuda

echo
cat "$RUN_ROOT/evals/debug_ddim_current/summary.json"
