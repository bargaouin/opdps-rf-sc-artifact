#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from opdps_rf_sc.cache import FrameCache
from opdps_rf_sc.data import normalize_unit_power
from opdps_rf_sc.constants import OFFICIAL_MIXTURE_LENGTH, SIGNAL_TYPES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--windows-per-class", type=int, default=256)
    ap.add_argument("--length", type=int, default=OFFICIAL_MIXTURE_LENGTH)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    cache = FrameCache(args.cache_dir)
    rng = np.random.default_rng(args.seed)
    out = {}
    for sig in SIGNAL_TYPES:
        psd = np.zeros(args.length, dtype=np.float64)
        for _ in range(args.windows_per_class):
            x = cache.sample_window(sig, args.length, rng, split="train")
            x = normalize_unit_power(x)
            xf = np.fft.fft(x, norm="ortho")
            psd += np.abs(xf) ** 2
        psd /= args.windows_per_class
        psd = np.maximum(psd, 1e-8)
        # Mild circular smoothing to reduce single-bin variance.
        k = 33
        kernel = np.ones(k) / k
        pad = k // 2
        ext = np.concatenate([psd[-pad:], psd, psd[:pad]])
        psd = np.convolve(ext, kernel, mode="valid")
        psd /= psd.mean()
        out[sig] = psd.astype(np.float32)
        print(f"{sig}: mean PSD={psd.mean():.6f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **out)


if __name__ == "__main__":
    main()
