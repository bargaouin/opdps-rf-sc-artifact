#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from opdps_rf_sc.config import load_yaml
from opdps_rf_sc.data import MixtureBatchGenerator
from opdps_rf_sc.model_io import load_coarse_from_checkpoint
from opdps_rf_sc.train_utils import configure_torch, resolve_device


def complex_mse_ri(pred, target):
    return (
        (pred - target)
        .square()
        .sum(dim=1)
        .mean(dim=1)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--coarse-checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    configure_torch()
    device = resolve_device(args.device)

    coarse, _ = load_coarse_from_checkpoint(
        args.coarse_checkpoint,
        device,
        use_ema=True,
    )

    tcfg = cfg["train"]

    gen = MixtureBatchGenerator(
        cache_dir=args.cache_dir,
        length=int(tcfg.get("length", 40960)),
        continuous_sinr_prob=float(tcfg.get("continuous_sinr_prob", 0.2)),
        continuous_sinr_range=(
            float(tcfg.get("continuous_sinr_min", -15)),
            float(tcfg.get("continuous_sinr_max", 5)),
        ),
        low_sinr_bias=float(tcfg.get("low_sinr_bias", 0.35)),
        common_phase_aug_prob=0.0,
        split="holdout",
        seed=616161,
    )

    vals = []

    with torch.no_grad():
        for i in range(args.batches):
            batch = gen.batch(args.batch_size, device)
            pred = coarse(batch.mixture, batch.class_id).float()

            mse = complex_mse_ri(
                pred,
                batch.target.float(),
            )

            vals.extend(mse.cpu().tolist())

            print(
                f"batch={i+1}/{args.batches} "
                f"mean={mse.mean().item():.8f}"
            )

    print()
    print(
        "MATCHED_COARSE_HOLDOUT_COMPLEX_MSE="
        f"{sum(vals)/len(vals):.10f}"
    )


if __name__ == "__main__":
    main()
