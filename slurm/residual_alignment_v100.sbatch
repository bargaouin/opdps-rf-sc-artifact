#!/usr/bin/env bash
#SBATCH --job-name=rfc-align
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

CONFIG="$REPO/configs/proposed_v100.yaml"
COARSE="$RUN_ROOT/proposed_v100/coarse/stage1_final_best.pt"
DIFF="$RUN_ROOT/proposed_v100/operator_diffusion/best.pt"
RSTATS="$RUN_ROOT/stage2_shared/residual_stats.json"
COARSE_METRICS="$RUN_ROOT/evals/final_coarse_v100/metrics.csv"
OUT="$RUN_ROOT/evals/residual_alignment_v100"

mkdir -p "$OUT"

echo "===== DEVICE ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "===== RESIDUAL ALIGNMENT ====="

python scripts/eval_residual_alignment.py \
    --cache-dir "$CACHE_DIR" \
    --config "$CONFIG" \
    --coarse-checkpoint "$COARSE" \
    --diffusion-checkpoint "$DIFF" \
    --residual-stats "$RSTATS" \
    --coarse-metrics "$COARSE_METRICS" \
    --output-dir "$OUT" \
    --sinr 3.0 \
    --per-class 32 \
    --batch-size 4 \
    --solver-steps 20 \
    --sampler dpm \
    --k-values 1 4 8 \
    --betas 0 0.005 0.01 0.02 0.05 0.10 0.20 0.50 1.0 \
    --seed 1337 \
    --device cuda

echo
echo "===== SUMMARY.JSON ====="
cat "$OUT/summary.json"

echo
echo "===== BETA SWEEP ====="
cat "$OUT/beta_sweep_summary.csv"

echo
echo "===== BY CLASS ====="
cat "$OUT/alignment_by_class.csv"
