#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opdps_rf_sc.cache import build_frame_cache, build_sep_val_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-root", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--holdout-last", type=int, default=50,
                    help="Last frames per class are excluded from training because official sep_val was generated from them.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-sep-val", action="store_true", help="Skip caching the 2200 official sep_val examples")
    args = ap.parse_args()
    manifest = build_frame_cache(
        args.rf_root, args.cache_dir, holdout_last=args.holdout_last, overwrite=args.overwrite
    )
    if not args.skip_sep_val:
        manifest = build_sep_val_cache(args.rf_root, args.cache_dir, overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
