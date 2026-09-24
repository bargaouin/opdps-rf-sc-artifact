#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from opdps_rf_sc.cache import SepValCache
from opdps_rf_sc.constants import ID_TO_INTERFERENCE, OFFICIAL_SINR_LEVELS_DESC
from opdps_rf_sc.metrics import mse_complex_np, nmse_db_complex_np


def sinr_label(idx):
    return float(OFFICIAL_SINR_LEVELS_DESC[idx // 100]) if idx // 100 < 11 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=["mixture", "psd_wiener"], required=True)
    ap.add_argument("--psd-templates", default=None)
    args = ap.parse_args()

    cache = SepValCache(args.cache_dir)
    templates = None
    if args.method == "psd_wiener":
        if not args.psd_templates:
            raise ValueError("--psd-templates is required for psd_wiener")
        templates = np.load(args.psd_templates)

    rows = []
    for row in range(cache.mixture.shape[0]):
        y = np.asarray(cache.mixture[row], dtype=np.complex64)
        s = np.asarray(cache.target[row], dtype=np.complex64)
        cid = int(cache.class_id[row])
        idx = int(cache.idx[row])
        intr = ID_TO_INTERFERENCE[cid]
        if args.method == "mixture":
            shat = y
        else:
            ps = np.asarray(templates["CommSignal2"], dtype=np.float64)
            pb = np.asarray(templates[intr], dtype=np.float64)
            # E|y|^2 ~= 1 + a because the official generator normalizes the SOI to unit power.
            a = max(float(np.mean(np.abs(y) ** 2)) - 1.0, 1e-4)
            h = ps / (ps + a * pb + 1e-8)
            shat = np.fft.ifft(np.fft.fft(y) * h).astype(np.complex64)
        rows.append({
            "method": args.method,
            "interference": intr,
            "idx": idx,
            "sinr_db": sinr_label(idx),
            "mse_soi": mse_complex_np(shat, s),
            "nmse_soi_db": nmse_db_complex_np(shat, s),
            "mse_interference": mse_complex_np(y - shat, y - s),
        })

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "metrics.csv", index=False)
    by = df.groupby(["interference", "sinr_db"], as_index=False).agg(
        mse_soi=("mse_soi", "mean"), nmse_soi_db=("nmse_soi_db", "mean"), n=("idx", "count")
    )
    by.to_csv(outdir / "summary_by_sinr.csv", index=False)
    summary = {
        "method": args.method,
        "examples": len(df),
        "mean_mse_soi": float(df.mse_soi.mean()),
        "mean_nmse_soi_db": float(df.nmse_soi_db.mean()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
