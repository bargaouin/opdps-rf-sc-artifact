#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from opdps_rf_sc.data import MixtureBatchGenerator
from opdps_rf_sc.diffusion import DiffusionSchedule, sample_residual_ensemble
from opdps_rf_sc.model_io import load_coarse_from_checkpoint, load_diffusion_from_checkpoint
from opdps_rf_sc.train_utils import configure_torch, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--coarse-checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--diffusion-checkpoint", default=None)
    ap.add_argument("--residual-stats", default=None)
    ap.add_argument("--lengths", default="4096,8192,16384,32768,40960")
    ap.add_argument("--samples-per-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sampler", choices=["dpm", "ddim"], default="dpm")
    ap.add_argument("--solver-steps", type=int, default=20)
    ap.add_argument("--posterior-samples", type=int, default=2)
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    configure_torch()
    device = resolve_device(args.device)
    coarse, cckpt = load_coarse_from_checkpoint(args.coarse_checkpoint, device, use_ema=True)
    diffusion = schedule = scales = None
    if args.diffusion_checkpoint:
        diffusion, dckpt = load_diffusion_from_checkpoint(args.diffusion_checkpoint, device, use_ema=True)
        stats = json.loads(Path(args.residual_stats).read_text())
        scales = torch.tensor(stats["scales"], dtype=torch.float32, device=device)
        schedule = DiffusionSchedule(int(dckpt["config"]["diffusion"].get("timesteps", 1000))).to(device)

    rows = []
    for length in [int(x) for x in args.lengths.split(",") if x.strip()]:
        gen = MixtureBatchGenerator(
            cache_dir=args.cache_dir, split="holdout", length=length,
            continuous_sinr_prob=0.0, low_sinr_bias=0.0, seed=1000 + length
        )
        remaining = args.samples_per_length
        seen = 0
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            batch = gen.batch(bs, device)
            with torch.no_grad():
                c = coarse(batch.mixture, batch.class_id).float()
                p = c
                if diffusion is not None:
                    p = sample_residual_ensemble(
                        diffusion, batch.mixture, c, batch.class_id, schedule, scales,
                        num_samples=args.posterior_samples, steps=args.solver_steps,
                        sampler=args.sampler, blend=args.blend, seed=length * 1000 + seen,
                    )
                mse = (p - batch.target).square().sum(dim=1).mean(dim=1)
            for j in range(bs):
                rows.append({
                    "length": length,
                    "mse": float(mse[j].cpu()),
                    "sinr_db": float(batch.sinr_db[j].cpu()),
                    "class_id": int(batch.class_id[j].cpu()),
                })
            remaining -= bs
            seen += bs
        print(f"length={length}: done {seen}")

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = df.groupby("length", as_index=False).agg(mean_mse=("mse", "mean"), median_mse=("mse", "median"), n=("mse", "count"))
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
