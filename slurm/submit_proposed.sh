#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}" "${CACHE_DIR:?}" "${RUN_ROOT:?}" "${VENV:?}"
mkdir -p "$REPO/logs" "$RUN_ROOT/proposed"
cd "$REPO"
CONFIG="$REPO/configs/proposed.yaml"

DEP=""
if [[ ! -f "$CACHE_DIR/manifest.json" ]] || ! grep -q '"sep_val"' "$CACHE_DIR/manifest.json"; then
  : "${RF_ROOT:?Need RF_ROOT to build cache}" 
  J_CACHE=$(sbatch --parsable --export=ALL,REPO="$REPO",RF_ROOT="$RF_ROOT",CACHE_DIR="$CACHE_DIR",RF_ENV="${RF_ENV:-}" "$REPO/slurm/prepare_cache.sbatch")
  DEP="--dependency=afterok:$J_CACHE"
  echo "cache job: $J_CACHE"
fi

COARSE_DIR="$RUN_ROOT/proposed/coarse"
J_COARSE=$(sbatch --parsable $DEP --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",RUN_DIR="$COARSE_DIR",CONFIG="$CONFIG",VENV="$VENV" "$REPO/slurm/train_coarse.sbatch")
echo "coarse job: $J_COARSE"

STATS="$RUN_ROOT/proposed/residual_stats.json"
J_STATS=$(sbatch --parsable --dependency=afterok:$J_COARSE --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",CONFIG="$CONFIG",COARSE_CKPT="$COARSE_DIR/best.pt",STATS_OUT="$STATS",VENV="$VENV" "$REPO/slurm/residual_stats.sbatch")
echo "residual stats job: $J_STATS"

DIFF_DIR="$RUN_ROOT/proposed/diffusion"
J_DIFF=$(sbatch --parsable --dependency=afterok:$J_STATS --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",RUN_DIR="$DIFF_DIR",CONFIG="$CONFIG",COARSE_CKPT="$COARSE_DIR/best.pt",STATS_OUT="$STATS",VENV="$VENV" "$REPO/slurm/train_diffusion.sbatch")
echo "diffusion job: $J_DIFF"

BLEND="$RUN_ROOT/proposed/blend.json"
J_BLEND=$(sbatch --parsable --dependency=afterok:$J_DIFF --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",CONFIG="$CONFIG",COARSE_CKPT="$COARSE_DIR/best.pt",DIFF_CKPT="$DIFF_DIR/best.pt",STATS_OUT="$STATS",BLEND_OUT="$BLEND",VENV="$VENV" "$REPO/slurm/tune_blend.sbatch")
echo "blend job: $J_BLEND"

EVAL="$RUN_ROOT/proposed/eval"
J_EVAL=$(sbatch --parsable --dependency=afterok:$J_BLEND --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",CONFIG="$CONFIG",COARSE_CKPT="$COARSE_DIR/best.pt",DIFF_CKPT="$DIFF_DIR/best.pt",STATS_OUT="$STATS",BLEND_OUT="$BLEND",EVAL_OUT="$EVAL",VENV="$VENV" "$REPO/slurm/eval_sep_val.sbatch")
echo "official sep_val evaluation job: $J_EVAL"

echo "Pipeline submitted. Final results will be in $EVAL/summary.json and summary_by_sinr.csv"
