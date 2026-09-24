#!/usr/bin/env bash
#SBATCH --job-name=rfc-coarse
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${REPO:?}" "${CACHE_DIR:?}" "${RUN_DIR:?}" "${CONFIG:?}" "${VENV:?}"
mkdir -p "$REPO/logs" "$RUN_DIR"
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
ARGS=(--config "$CONFIG" --cache-dir "$CACHE_DIR" --run-dir "$RUN_DIR")
[[ -n "${SEED:-}" ]] && ARGS+=(--seed "$SEED")
[[ -n "${RESUME:-}" ]] && ARGS+=(--resume "$RESUME")
python scripts/train_coarse.py "${ARGS[@]}"
