#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from opdps_rf_sc.checkpoint import load_checkpoint, save_checkpoint
from opdps_rf_sc.config import load_yaml
from opdps_rf_sc.data import MixtureBatchGenerator
from opdps_rf_sc.diffusion import DiffusionSchedule
from opdps_rf_sc.ema import EMA
from opdps_rf_sc.model_io import load_coarse_from_checkpoint
from opdps_rf_sc.models import build_separator
from opdps_rf_sc.train_utils import (
    autocast_context, configure_torch, count_parameters, make_lr_scheduler, resolve_device, seed_all
)


def load_scales(path, device):
    obj = json.loads(Path(path).read_text())
    return torch.tensor(obj["scales"], dtype=torch.float32, device=device)


@torch.no_grad()
def validate_noise_loss(model, coarse, gen, scales, schedule, batches, batch_size, device, amp):
    model.eval()
    vals = []
    for _ in range(batches):
        batch = gen.batch(batch_size, device)
        with autocast_context(device, amp):
            c = coarse(batch.mixture, batch.class_id)
        scale = scales[batch.class_id].reshape(-1, 1, 1)
        x0 = (batch.target - c.float()) / scale
        t = torch.randint(0, schedule.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        xt = schedule.add_noise(x0, noise, t)
        inp = torch.cat([xt, batch.mixture, c.float()], dim=1)
        with autocast_context(device, amp):
            eps = model(inp, batch.class_id, t=t.float())
        vals.append(F.mse_loss(eps.float(), noise).item())
    model.train()
    return sum(vals) / max(1, len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--coarse-checkpoint", required=True)
    ap.add_argument("--residual-stats", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg = dict(cfg); cfg["seed"] = int(args.seed)
    dcfg = cfg["diffusion"]
    if args.steps is not None:
        dcfg = dict(dcfg)
        dcfg["steps"] = args.steps
        cfg = dict(cfg)
        cfg["diffusion"] = dcfg

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2))

    seed_all(int(cfg.get("seed", 1337)) + 101)
    configure_torch()
    device = resolve_device(args.device)
    amp = bool(dcfg.get("amp_bf16", True))

    coarse, _ = load_coarse_from_checkpoint(args.coarse_checkpoint, device, use_ema=True)
    for p in coarse.parameters():
        p.requires_grad_(False)

    model = build_separator(dcfg["model"], in_channels=6, out_channels=2, time_conditioned=True).to(device)
    ema = EMA(model, decay=float(dcfg.get("ema_decay", 0.9999)))
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(dcfg["lr"]), weight_decay=float(dcfg.get("weight_decay", 1e-4)), betas=(0.9, 0.99)
    )
    total_steps = int(dcfg["steps"])
    sched_lr = make_lr_scheduler(
        opt, total_steps=total_steps, warmup_steps=int(dcfg.get("warmup_steps", 0)),
        min_ratio=float(dcfg.get("min_lr_ratio", 0.05))
    )
    schedule = DiffusionSchedule(int(dcfg.get("timesteps", 1000))).to(device)
    scales = load_scales(args.residual_stats, device)

    start_step = 0
    best = float("inf")
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])
        if ckpt.get("optimizer"):
            opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            sched_lr.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        best = float(ckpt.get("extra", {}).get("best_val_noise_mse", best))

    tcfg = cfg["train"]
    common_data = dict(
        cache_dir=args.cache_dir,
        length=int(tcfg.get("length", 40960)),
        continuous_sinr_prob=float(tcfg.get("continuous_sinr_prob", 0.2)),
        continuous_sinr_range=(float(tcfg.get("continuous_sinr_min", -15)), float(tcfg.get("continuous_sinr_max", 5))),
        low_sinr_bias=float(tcfg.get("low_sinr_bias", 0.35)),
        common_phase_aug_prob=float(tcfg.get("common_phase_aug_prob", 0.5)),
    )
    train_gen = MixtureBatchGenerator(**common_data, split="train", seed=int(cfg.get("seed", 1337)) + 111)
    val_data = dict(common_data); val_data["common_phase_aug_prob"] = 0.0
    val_gen = MixtureBatchGenerator(**val_data, split="holdout", seed=515151)

    print(f"device={device} diffusion_params={count_parameters(model):,} residual_scales={scales.tolist()}")
    batch_size = int(dcfg["batch_size"])
    x0_weight = float(dcfg.get("x0_weight", 0.05))
    model.train()
    last = time.time()

    for step in range(start_step + 1, total_steps + 1):
        batch = train_gen.batch(batch_size, device)
        with torch.no_grad(), autocast_context(device, amp):
            coarse_pred = coarse(batch.mixture, batch.class_id)
        coarse_pred = coarse_pred.float()
        scale = scales[batch.class_id].reshape(-1, 1, 1)
        x0 = (batch.target - coarse_pred) / scale
        t = torch.randint(0, schedule.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        xt = schedule.add_noise(x0, noise, t)

        opt.zero_grad(set_to_none=True)
        inp = torch.cat([xt, batch.mixture, coarse_pred], dim=1)
        with autocast_context(device, amp):
            eps = model(inp, batch.class_id, t=t.float())
            eps_loss = F.mse_loss(eps, noise)
            if x0_weight > 0:
                x0_pred = schedule.predict_x0(xt.float(), eps.float(), t)
                # Emphasize reconstruction only where x0 is numerically identifiable.
                a = schedule.alpha_bar[t].reshape(-1, 1, 1)
                rec_weight = a.detach()
                x0_loss = ((x0_pred - x0).square() * rec_weight).mean()
            else:
                x0_loss = eps_loss.new_zeros(())
            loss = eps_loss + x0_weight * x0_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(dcfg.get("grad_clip", 1.0)))
        opt.step()
        sched_lr.step()
        ema.update(model)

        if step % 100 == 0 or step == 1:
            dt = time.time() - last
            last = time.time()
            print(
                f"step={step:07d} loss={float(loss.detach()):.6g} eps={float(eps_loss.detach()):.6g} "
                f"x0={float(x0_loss.detach()):.6g} lr={sched_lr.get_last_lr()[0]:.3e} sec/100={dt:.2f}",
                flush=True,
            )

        if step % int(dcfg.get("validate_every", 2500)) == 0 or step == total_steps:
            v = validate_noise_loss(
                ema.shadow, coarse, val_gen, scales, schedule,
                int(dcfg.get("val_batches", 8)), batch_size, device, amp
            )
            print(f"HOLDOUT diffusion_noise_mse={v:.8g} at step={step}", flush=True)
            save_checkpoint(
                run_dir / "latest.pt", model=model, ema=ema, optimizer=opt, scheduler=sched_lr,
                step=step, config=cfg,
                extra={
                    "best_val_noise_mse": min(best, v),
                    "val_noise_mse": v,
                    "coarse_checkpoint": str(Path(args.coarse_checkpoint).resolve()),
                    "residual_stats": str(Path(args.residual_stats).resolve()),
                },
            )
            if v < best:
                best = v
                save_checkpoint(
                    run_dir / "best.pt", model=model, ema=ema, optimizer=opt, scheduler=sched_lr,
                    step=step, config=cfg,
                    extra={
                        "best_val_noise_mse": best,
                        "val_noise_mse": v,
                        "coarse_checkpoint": str(Path(args.coarse_checkpoint).resolve()),
                        "residual_stats": str(Path(args.residual_stats).resolve()),
                    },
                )
        elif step % int(dcfg.get("save_every", 5000)) == 0:
            save_checkpoint(
                run_dir / "latest.pt", model=model, ema=ema, optimizer=opt, scheduler=sched_lr,
                step=step, config=cfg,
                extra={
                    "best_val_noise_mse": best,
                    "coarse_checkpoint": str(Path(args.coarse_checkpoint).resolve()),
                    "residual_stats": str(Path(args.residual_stats).resolve()),
                },
            )


if __name__ == "__main__":
    main()
