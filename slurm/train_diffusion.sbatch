#!/usr/bin/env bash
#SBATCH --job-name=rfc-diff
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${REPO:?}" "${CACHE_DIR:?}" "${RUN_DIR:?}" "${CONFIG:?}" "${COARSE_CKPT:?}" "${STATS_OUT:?}" "${VENV:?}"
mkdir -p "$REPO/logs" "$RUN_DIR"
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
ARGS=(--config "$CONFIG" --cache-dir "$CACHE_DIR" --coarse-checkpoint "$COARSE_CKPT" \
      --residual-stats "$STATS_OUT" --run-dir "$RUN_DIR")
[[ -n "${SEED:-}" ]] && ARGS+=(--seed "$SEED")
[[ -n "${RESUME:-}" ]] && ARGS+=(--resume "$RESUME")
python scripts/train_diffusion.py "${ARGS[@]}"
