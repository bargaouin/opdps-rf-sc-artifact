#!/usr/bin/env bash
#SBATCH --job-name=rfc-length
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${REPO:?}" "${CACHE_DIR:?}" "${COARSE_CKPT:?}" "${LENGTH_OUT:?}" "${VENV:?}"
mkdir -p "$REPO/logs" "$(dirname "$LENGTH_OUT")"
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
ARGS=(--cache-dir "$CACHE_DIR" --coarse-checkpoint "$COARSE_CKPT" --output "$LENGTH_OUT" \
      --lengths "${LENGTHS:-4096,8192,16384,32768,40960}" --samples-per-length "${N_PER_LENGTH:-256}" --batch-size "${BATCH_SIZE:-4}")
if [[ -n "${DIFF_CKPT:-}" ]]; then
  ARGS+=(--diffusion-checkpoint "$DIFF_CKPT" --residual-stats "$STATS_OUT" \
         --sampler "${SAMPLER:-dpm}" --solver-steps "${SOLVER_STEPS:-20}" --posterior-samples "${POSTERIOR_SAMPLES:-2}" --blend "${BLEND:-1.0}")
fi
python scripts/eval_length_generalization.py "${ARGS[@]}"
