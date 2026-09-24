#!/usr/bin/env bash
#SBATCH --job-name=rfc-blend
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /path/to/opdps_rf_sc
source slurm/env.sh

V100_VENV=/path/to/opdps-rf-sc-v100
source "$V100_VENV/bin/activate"

CONFIG="$REPO/configs/proposed_v100.yaml"

COARSE_CKPT="$RUN_ROOT/proposed_v100/coarse/stage1_final_best.pt"
DIFF_CKPT="$RUN_ROOT/proposed_v100/operator_diffusion/best.pt"
RSTATS="$RUN_ROOT/stage2_shared/residual_stats.json"

for B in 0.05 0.10 0.20 0.30 0.50 0.75 1.00
do
    OUT="$RUN_ROOT/evals/debug_blend_${B}"
    mkdir -p "$OUT"

    echo
    echo "========================================"
    echo "BLEND = $B"
    echo "========================================"

    python scripts/eval_sep_val.py \
        --cache-dir "$CACHE_DIR" \
        --config "$CONFIG" \
        --coarse-checkpoint "$COARSE_CKPT" \
        --diffusion-checkpoint "$DIFF_CKPT" \
        --residual-stats "$RSTATS" \
        --output-dir "$OUT" \
        --sampler dpm \
        --solver-steps 20 \
        --posterior-samples 1 \
        --blend "$B" \
        --batch-size 4 \
        --max-examples 32 \
        --device cuda

    echo "SUMMARY blend=$B"
    cat "$OUT/summary.json"
done
