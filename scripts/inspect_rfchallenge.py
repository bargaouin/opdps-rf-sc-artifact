#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from opdps_rf_sc.constants import OFFICIAL_TRAIN_FRAME_COUNTS, SIGNAL_TYPES, SEPARATION_INTERFERERS
from opdps_rf_sc.rfchallenge import RFCAdapter, validate_official_separation_example


def measured_sinr_db_np(signal: np.ndarray, interference: np.ndarray) -> float:
    ps = float(np.mean(np.abs(signal) ** 2))
    pi = float(np.mean(np.abs(interference) ** 2))
    return 10.0 * np.log10(max(ps, 1e-30) / max(pi, 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-root", required=True, help="Inner rfchallenge_singlechannel_starter-main folder containing dataset/")
    args = ap.parse_args()

    adapter = RFCAdapter(args.rf_root)
    report = {"rf_root": str(Path(args.rf_root).expanduser().resolve()), "train_frame": {}, "generated": {}, "sep_val": {}}

    for sig in SIGNAL_TYPES:
        x, meta = adapter.load_sample(0, "train_frame", sig)
        x = np.asarray(x)
        report["train_frame"][sig] = {
            "official_count": OFFICIAL_TRAIN_FRAME_COUNTS[sig],
            "first_frame_shape": list(x.shape),
            "dtype": str(x.dtype),
            "power": float(np.mean(np.abs(x) ** 2)),
        }

    for intr in SEPARATION_INTERFERERS:
        y, s, b = adapter.create_sep_mixture(intr, target_sinr_db=-6.0, dataset_type="train")
        y, s, b = map(np.asarray, (y, s, b))
        report["generated"][intr] = {
            "shape": list(y.shape),
            "additivity_max_abs_error": float(np.max(np.abs(y - (s + b)))),
            "measured_sinr_db": measured_sinr_db_np(s, b),
        }

    for intr in SEPARATION_INTERFERERS:
        y, s, b, *_ = validate_official_separation_example(adapter, intr, idx=401)
        report["sep_val"][intr] = {
            "shape": list(y.shape),
            "measured_sinr_db_idx401": measured_sinr_db_np(s, b),
            "additivity_max_abs_error": float(np.max(np.abs(y - (s + b)))),
        }

    print(json.dumps(report, indent=2))
    print("\nPASS: official rfcutils can load train_frame, generate mixtures, and read sep_val/components.")


if __name__ == "__main__":
    main()
