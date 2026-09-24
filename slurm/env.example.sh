# Copy this to slurm/env.sh and edit paths for Athena.
export REPO=/home/user/opdps_rf_sc
# IMPORTANT: point to the INNER folder that contains dataset/, notebook/, rfcutils/.
export RF_ROOT=/home/user/rfchallenge_singlechannel_starter/rfchallenge_singlechannel_starter-main
export CACHE_DIR=/home/user/rfchallenge_cache_sc
export RUN_ROOT=/home/user/runs/opdps_rf_sc
export VENV=/home/user/venvs/opdps-rf-sc
# Environment used only while decoding/caching the legacy SigMF dataset.
# Do NOT copy a macOS RF-env to Athena; create a Linux one with scripts/setup_rfchallenge_env.sh.
export RF_ENV=/home/user/venvs/rfchallenge-2021
