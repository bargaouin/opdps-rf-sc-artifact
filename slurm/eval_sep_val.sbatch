#!/usr/bin/env bash
#SBATCH --job-name=rfc-eval
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${REPO:?}" "${CACHE_DIR:?}" "${CONFIG:?}" "${COARSE_CKPT:?}" "${EVAL_OUT:?}" "${VENV:?}"
mkdir -p "$REPO/logs" "$EVAL_OUT"
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
ARGS=(--cache-dir "$CACHE_DIR" --config "$CONFIG" --coarse-checkpoint "$COARSE_CKPT" \
      --output-dir "$EVAL_OUT" --batch-size "${BATCH_SIZE:-4}" --resume)
if [[ -n "${DIFF_CKPT:-}" ]]; then
  ARGS+=(--diffusion-checkpoint "$DIFF_CKPT" --residual-stats "$STATS_OUT")
  [[ -n "${BLEND_OUT:-}" ]] && ARGS+=(--blend-json "$BLEND_OUT")
  ARGS+=(--sampler "${SAMPLER:-dpm}" --solver-steps "${SOLVER_STEPS:-20}" --posterior-samples "${POSTERIOR_SAMPLES:-4}")
fi
python scripts/eval_sep_val.py "${ARGS[@]}"
