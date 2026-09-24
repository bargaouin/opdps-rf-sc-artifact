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

from opdps_rf_sc.checkpoint import load_checkpoint, save_checkpoint
from opdps_rf_sc.config import load_yaml
from opdps_rf_sc.data import MixtureBatchGenerator
from opdps_rf_sc.ema import EMA
from opdps_rf_sc.losses import separation_loss
from opdps_rf_sc.metrics import mse_2ch_torch
from opdps_rf_sc.models import build_separator
from opdps_rf_sc.train_utils import (
    autocast_context, configure_torch, count_parameters, make_lr_scheduler, resolve_device, seed_all
)


@torch.no_grad()
def validate(model, generator, batches, batch_size, device, amp):
    model.eval()
    total = 0.0
    n = 0
    for _ in range(batches):
        batch = generator.batch(batch_size, device)
        with autocast_context(device, amp):
            pred = model(batch.mixture, batch.class_id)
        m = mse_2ch_torch(pred.float(), batch.target.float())
        total += float(m.sum().cpu())
        n += int(m.numel())
    model.train()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="Override config steps (useful for smoke tests)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg = dict(cfg); cfg["seed"] = int(args.seed)
    tcfg = cfg["train"]
    if args.steps is not None:
        tcfg = dict(tcfg)
        tcfg["steps"] = args.steps
        cfg = dict(cfg)
        cfg["train"] = tcfg

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2))

    seed_all(int(cfg.get("seed", 1337)))
    configure_torch()
    device = resolve_device(args.device)
    amp = bool(tcfg.get("amp_bf16", True))

    model = build_separator(cfg["model"], in_channels=2, out_channels=2, time_conditioned=False).to(device)
    ema = EMA(model, decay=float(tcfg.get("ema_decay", 0.9999)))
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 1e-4)), betas=(0.9, 0.99)
    )
    total_steps = int(tcfg["steps"])
    sched = make_lr_scheduler(
        opt, total_steps=total_steps, warmup_steps=int(tcfg.get("warmup_steps", 0)),
        min_ratio=float(tcfg.get("min_lr_ratio", 0.05))
    )
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
            sched.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        best = float(ckpt.get("extra", {}).get("best_val_mse", best))

    common_data = dict(
        cache_dir=args.cache_dir,
        length=int(tcfg.get("length", 40960)),
        continuous_sinr_prob=float(tcfg.get("continuous_sinr_prob", 0.2)),
        continuous_sinr_range=(float(tcfg.get("continuous_sinr_min", -15)), float(tcfg.get("continuous_sinr_max", 5))),
        low_sinr_bias=float(tcfg.get("low_sinr_bias", 0.35)),
        common_phase_aug_prob=float(tcfg.get("common_phase_aug_prob", 0.5)),
    )
    train_gen = MixtureBatchGenerator(**common_data, split="train", seed=int(cfg.get("seed", 1337)))
    val_data = dict(common_data); val_data["common_phase_aug_prob"] = 0.0
    val_gen = MixtureBatchGenerator(**val_data, split="holdout", seed=424242)

    print(f"device={device} params={count_parameters(model):,} start_step={start_step} total_steps={total_steps}")
    model.train()
    last = time.time()
    batch_size = int(tcfg["batch_size"])
    for step in range(start_step + 1, total_steps + 1):
        batch = train_gen.batch(batch_size, device)
        opt.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            pred = model(batch.mixture, batch.class_id)
            loss, parts = separation_loss(pred, batch.target, tcfg.get("loss", {}))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 1.0)))
        opt.step()
        sched.step()
        ema.update(model)

        if step % 100 == 0 or step == 1:
            dt = time.time() - last
            last = time.time()
            print(
                f"step={step:07d} loss={parts['loss']:.6g} mse={parts['mse']:.6g} "
                f"lr={sched.get_last_lr()[0]:.3e} sec/100={dt:.2f}", flush=True
            )

        if step % int(tcfg.get("validate_every", 2000)) == 0 or step == total_steps:
            v = validate(
                ema.shadow, val_gen, int(tcfg.get("val_batches", 8)), batch_size, device, amp
            )
            print(f"HOLDOUT val_mse={v:.8g} at step={step}", flush=True)
            extra = {"best_val_mse": min(best, v), "val_mse": v}
            save_checkpoint(
                run_dir / "latest.pt", model=model, ema=ema, optimizer=opt, scheduler=sched,
                step=step, config=cfg, extra=extra
            )
            if v < best:
                best = v
                save_checkpoint(
                    run_dir / "best.pt", model=model, ema=ema, optimizer=opt, scheduler=sched,
                    step=step, config=cfg, extra={"best_val_mse": best, "val_mse": v}
                )
        elif step % int(tcfg.get("save_every", 5000)) == 0:
            save_checkpoint(
                run_dir / "latest.pt", model=model, ema=ema, optimizer=opt, scheduler=sched,
                step=step, config=cfg, extra={"best_val_mse": best}
            )


if __name__ == "__main__":
    main()
