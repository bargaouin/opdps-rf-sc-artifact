#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from opdps_rf_sc.config import load_yaml
from opdps_rf_sc.constants import ID_TO_INTERFERENCE
from opdps_rf_sc.data import MixtureBatchGenerator
from opdps_rf_sc.model_io import load_coarse_from_checkpoint
from opdps_rf_sc.train_utils import autocast_context, configure_torch, resolve_device, seed_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--coarse-checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]
    seed_all(int(cfg.get("seed", 1337)) + 19)
    configure_torch()
    device = resolve_device(args.device)
    coarse, _ = load_coarse_from_checkpoint(args.coarse_checkpoint, device, use_ema=True)

    gen = MixtureBatchGenerator(
        cache_dir=args.cache_dir,
        split="train",
        length=int(tcfg.get("length", 40960)),
        continuous_sinr_prob=float(tcfg.get("continuous_sinr_prob", 0.2)),
        continuous_sinr_range=(float(tcfg.get("continuous_sinr_min", -15)), float(tcfg.get("continuous_sinr_max", 5))),
        low_sinr_bias=float(tcfg.get("low_sinr_bias", 0.35)),
        common_phase_aug_prob=float(tcfg.get("common_phase_aug_prob", 0.5)),
        seed=99123,
    )

    sums = torch.zeros(2, device=device, dtype=torch.float64)
    counts = torch.zeros(2, device=device, dtype=torch.float64)
    remaining = int(args.samples)
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        batch = gen.batch(bs, device)
        with torch.no_grad(), autocast_context(device, True):
            pred = coarse(batch.mixture, batch.class_id)
        residual = (batch.target - pred.float()).float()
        # Per-real-dimension variance. residual has [B,2,L].
        per_example_ss = residual.square().mean(dim=(1, 2)).double()
        for cid in (0, 1):
            m = batch.class_id == cid
            if m.any():
                sums[cid] += per_example_ss[m].sum()
                counts[cid] += int(m.sum())
        remaining -= bs

    var = sums / counts.clamp_min(1)
    scales = torch.sqrt(var).clamp_min(1e-3).cpu().tolist()
    out = {
        "coarse_checkpoint": str(Path(args.coarse_checkpoint).resolve()),
        "samples": int(args.samples),
        "scale_definition": "sqrt(mean((target-coarse)^2)) per real I/Q dimension",
        "scales": scales,
        "by_class": {
            ID_TO_INTERFERENCE[i]: {"class_id": i, "scale": scales[i], "count": int(counts[i].item())}
            for i in range(2)
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
