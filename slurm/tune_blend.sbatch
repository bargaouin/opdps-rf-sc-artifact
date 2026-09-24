#!/usr/bin/env bash
#SBATCH --job-name=rfc-blend
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${REPO:?}" "${CACHE_DIR:?}" "${CONFIG:?}" "${COARSE_CKPT:?}" "${DIFF_CKPT:?}" "${STATS_OUT:?}" "${BLEND_OUT:?}" "${VENV:?}"
mkdir -p "$REPO/logs" "$(dirname "$BLEND_OUT")"
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
python scripts/tune_blend.py --config "$CONFIG" --cache-dir "$CACHE_DIR" \
  --coarse-checkpoint "$COARSE_CKPT" --diffusion-checkpoint "$DIFF_CKPT" \
  --residual-stats "$STATS_OUT" --output "$BLEND_OUT" \
  --samples "${TUNE_SAMPLES:-512}" --batch-size "${BATCH_SIZE:-4}" \
  --solver-steps "${SOLVER_STEPS:-20}" --posterior-samples "${POSTERIOR_SAMPLES:-2}" --sampler "${SAMPLER:-dpm}"
