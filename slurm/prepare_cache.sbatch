#!/usr/bin/env bash
#SBATCH --job-name=rfc-cache
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${REPO:?}" "${RF_ROOT:?}" "${CACHE_DIR:?}"
mkdir -p "$REPO/logs" "$CACHE_DIR"
cd "$REPO"
if [[ -n "${RF_ENV:-}" && -f "$RF_ENV/bin/activate" ]]; then
  source "$RF_ENV/bin/activate"
fi
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
python scripts/inspect_rfchallenge.py --rf-root "$RF_ROOT"
python scripts/prepare_cache.py --rf-root "$RF_ROOT" --cache-dir "$CACHE_DIR"
