#!/usr/bin/env bash
#SBATCH --job-name=rfc-rstats
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${REPO:?}" "${CACHE_DIR:?}" "${CONFIG:?}" "${COARSE_CKPT:?}" "${STATS_OUT:?}" "${VENV:?}"
mkdir -p "$REPO/logs" "$(dirname "$STATS_OUT")"
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
python scripts/estimate_residual_stats.py --config "$CONFIG" --cache-dir "$CACHE_DIR" \
  --coarse-checkpoint "$COARSE_CKPT" --output "$STATS_OUT" --samples "${STATS_SAMPLES:-4096}" --batch-size "${BATCH_SIZE:-16}"
