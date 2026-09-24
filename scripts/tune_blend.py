#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from opdps_rf_sc.config import load_yaml
from opdps_rf_sc.data import MixtureBatchGenerator
from opdps_rf_sc.diffusion import DiffusionSchedule, sample_residual_ensemble
from opdps_rf_sc.model_io import load_coarse_from_checkpoint, load_diffusion_from_checkpoint
from opdps_rf_sc.train_utils import configure_torch, resolve_device, seed_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--coarse-checkpoint", required=True)
    ap.add_argument("--diffusion-checkpoint", required=True)
    ap.add_argument("--residual-stats", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--solver-steps", type=int, default=None)
    ap.add_argument("--posterior-samples", type=int, default=2)
    ap.add_argument("--sampler", default=None, choices=[None, "dpm", "ddim"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_all(91919)
    configure_torch()
    device = resolve_device(args.device)
    coarse, _ = load_coarse_from_checkpoint(args.coarse_checkpoint, device, use_ema=True)
    diffusion, dckpt = load_diffusion_from_checkpoint(args.diffusion_checkpoint, device, use_ema=True)
    stats = json.loads(Path(args.residual_stats).read_text())
    scales = torch.tensor(stats["scales"], dtype=torch.float32, device=device)
    schedule = DiffusionSchedule(int(cfg["diffusion"].get("timesteps", 1000))).to(device)
    icfg = cfg.get("inference", {})
    solver_steps = int(args.solver_steps or icfg.get("solver_steps", 20))
    sampler = args.sampler or icfg.get("sampler", "dpm")

    tcfg = cfg["train"]
    gen = MixtureBatchGenerator(
        cache_dir=args.cache_dir,
        split="holdout",
        length=int(tcfg.get("length", 40960)),
        continuous_sinr_prob=0.0,
        low_sinr_bias=0.0,
        common_phase_aug_prob=0.0,
        seed=818181,
    )

    grid = np.linspace(0.0, 1.5, 13)
    sums = np.zeros_like(grid)
    count = 0
    remaining = args.samples
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        batch = gen.batch(bs, device)
        with torch.no_grad():
            coarse_pred = coarse(batch.mixture, batch.class_id).float()
            # Get unit-blend residual posterior mean once, then evaluate all scalar blends cheaply.
            full = sample_residual_ensemble(
                diffusion, batch.mixture, coarse_pred, batch.class_id, schedule, scales,
                num_samples=args.posterior_samples, steps=solver_steps, sampler=sampler,
                blend=1.0, seed=100000 + count,
            )
            delta = full - coarse_pred
            for j, blend in enumerate(grid):
                pred = coarse_pred + float(blend) * delta
                mse = (pred - batch.target).square().sum(dim=1).mean(dim=1)
                sums[j] += float(mse.sum().cpu())
        count += bs
        remaining -= bs
        print(f"processed {count}/{args.samples}", flush=True)

    means = sums / max(count, 1)
    best_idx = int(np.argmin(means))
    result = {
        "blend_grid": grid.tolist(),
        "holdout_mse": means.tolist(),
        "best_blend": float(grid[best_idx]),
        "best_holdout_mse": float(means[best_idx]),
        "sampler": sampler,
        "solver_steps": solver_steps,
        "posterior_samples": args.posterior_samples,
        "samples": count,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
