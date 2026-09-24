#!/usr/bin/env bash
#SBATCH --job-name=rfc-length
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd /path/to/opdps_rf_sc
source slurm/env.sh

source /path/to/opdps-rf-sc-v100/bin/activate
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

NAMES=(
    "waveunet"
    "fno"
    "hybrid"
)

CKPTS=(
    "$RUN_ROOT/baseline_waveunet_v100/coarse/best.pt"
    "$RUN_ROOT/baseline_fno_v100/coarse/best.pt"
    "$RUN_ROOT/proposed_v100/coarse/stage1_final_best.pt"
)

IDX="$SLURM_ARRAY_TASK_ID"
NAME="${NAMES[$IDX]}"
CKPT="${CKPTS[$IDX]}"

OUTDIR="$RUN_ROOT/evals/length_generalization"
mkdir -p "$OUTDIR"

echo "===== MODEL: $NAME ====="
echo "checkpoint: $CKPT"

python scripts/eval_length_generalization.py \
    --cache-dir "$CACHE_DIR" \
    --coarse-checkpoint "$CKPT" \
    --output "$OUTDIR/${NAME}.csv" \
    --lengths "4096,8192,16384,32768,40960" \
    --samples-per-length 256 \
    --batch-size 8 \
    --device cuda

echo
echo "===== SUMMARY ====="
cat "$OUTDIR/${NAME}_summary.csv"
