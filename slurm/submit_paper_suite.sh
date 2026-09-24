#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}" "${CACHE_DIR:?}" "${RUN_ROOT:?}" "${VENV:?}"
if [[ ! -f "$CACHE_DIR/manifest.json" ]] || ! grep -q '"sep_val"' "$CACHE_DIR/manifest.json"; then
  echo "Cache/sep_val cache missing. Run slurm/submit_proposed.sh once or prepare_cache.sbatch first." >&2
  exit 2
fi
mkdir -p "$REPO/logs" "$RUN_ROOT"
cd "$REPO"

submit_coarse () {
  local tag="$1" cfg="$2"
  local dir="$RUN_ROOT/$tag/coarse"
  mkdir -p "$dir"
  sbatch --parsable --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",RUN_DIR="$dir",CONFIG="$cfg",VENV="$VENV" "$REPO/slurm/train_coarse.sbatch"
}
submit_direct_eval () {
  local dep="$1" tag="$2" cfg="$3"
  local dir="$RUN_ROOT/$tag"
  sbatch --parsable --dependency=afterok:"$dep" --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",CONFIG="$cfg",COARSE_CKPT="$dir/coarse/best.pt",EVAL_OUT="$dir/eval",VENV="$VENV" "$REPO/slurm/eval_sep_val.sbatch"
}

CFG_PROP="$REPO/configs/proposed.yaml"
CFG_WAVE="$REPO/configs/baseline_waveunet_direct.yaml"
CFG_FNO="$REPO/configs/baseline_fno_direct.yaml"
CFG_UNETD="$REPO/configs/baseline_unet_diffusion.yaml"

J_PROP=$(submit_coarse proposed "$CFG_PROP")
J_WAVE=$(submit_coarse waveunet_direct "$CFG_WAVE")
J_FNO=$(submit_coarse fno_direct "$CFG_FNO")
echo "coarse: proposed=$J_PROP waveunet=$J_WAVE fno=$J_FNO"

J_PROP_E=$(submit_direct_eval "$J_PROP" proposed "$CFG_PROP")
J_WAVE_E=$(submit_direct_eval "$J_WAVE" waveunet_direct "$CFG_WAVE")
J_FNO_E=$(submit_direct_eval "$J_FNO" fno_direct "$CFG_FNO")
echo "direct eval: proposed=$J_PROP_E waveunet=$J_WAVE_E fno=$J_FNO_E"

STATS="$RUN_ROOT/proposed/residual_stats.json"
J_STATS=$(sbatch --parsable --dependency=afterok:"$J_PROP" --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",CONFIG="$CFG_PROP",COARSE_CKPT="$RUN_ROOT/proposed/coarse/best.pt",STATS_OUT="$STATS",VENV="$VENV" "$REPO/slurm/residual_stats.sbatch")

# Proposed operator residual diffusion.
OPD_DIR="$RUN_ROOT/operator_diffusion/diffusion"; mkdir -p "$OPD_DIR"
J_OPD=$(sbatch --parsable --dependency=afterok:"$J_STATS" --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",RUN_DIR="$OPD_DIR",CONFIG="$CFG_PROP",COARSE_CKPT="$RUN_ROOT/proposed/coarse/best.pt",STATS_OUT="$STATS",VENV="$VENV" "$REPO/slurm/train_diffusion.sbatch")
OPD_BLEND="$RUN_ROOT/operator_diffusion/blend.json"
J_OPB=$(sbatch --parsable --dependency=afterok:"$J_OPD" --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",CONFIG="$CFG_PROP",COARSE_CKPT="$RUN_ROOT/proposed/coarse/best.pt",DIFF_CKPT="$OPD_DIR/best.pt",STATS_OUT="$STATS",BLEND_OUT="$OPD_BLEND",VENV="$VENV" "$REPO/slurm/tune_blend.sbatch")
J_OPE=$(sbatch --parsable --dependency=afterok:"$J_OPB" --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",CONFIG="$CFG_PROP",COARSE_CKPT="$RUN_ROOT/proposed/coarse/best.pt",DIFF_CKPT="$OPD_DIR/best.pt",STATS_OUT="$STATS",BLEND_OUT="$OPD_BLEND",EVAL_OUT="$RUN_ROOT/operator_diffusion/eval",VENV="$VENV" "$REPO/slurm/eval_sep_val.sbatch")

# Controlled ablation: same deterministic initializer, ordinary U-Net residual diffusion.
UND_DIR="$RUN_ROOT/unet_diffusion/diffusion"; mkdir -p "$UND_DIR"
J_UND=$(sbatch --parsable --dependency=afterok:"$J_STATS" --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",RUN_DIR="$UND_DIR",CONFIG="$CFG_UNETD",COARSE_CKPT="$RUN_ROOT/proposed/coarse/best.pt",STATS_OUT="$STATS",VENV="$VENV" "$REPO/slurm/train_diffusion.sbatch")
UND_BLEND="$RUN_ROOT/unet_diffusion/blend.json"
J_UNB=$(sbatch --parsable --dependency=afterok:"$J_UND" --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",CONFIG="$CFG_UNETD",COARSE_CKPT="$RUN_ROOT/proposed/coarse/best.pt",DIFF_CKPT="$UND_DIR/best.pt",STATS_OUT="$STATS",BLEND_OUT="$UND_BLEND",VENV="$VENV" "$REPO/slurm/tune_blend.sbatch")
J_UNE=$(sbatch --parsable --dependency=afterok:"$J_UNB" --export=ALL,REPO="$REPO",CACHE_DIR="$CACHE_DIR",CONFIG="$CFG_UNETD",COARSE_CKPT="$RUN_ROOT/proposed/coarse/best.pt",DIFF_CKPT="$UND_DIR/best.pt",STATS_OUT="$STATS",BLEND_OUT="$UND_BLEND",EVAL_OUT="$RUN_ROOT/unet_diffusion/eval",VENV="$VENV" "$REPO/slurm/eval_sep_val.sbatch")

echo "diffusion: operator train=$J_OPD eval=$J_OPE ; unet train=$J_UND eval=$J_UNE"
echo "Once these finish, run scripts/plot_results.py over the eval directories."
