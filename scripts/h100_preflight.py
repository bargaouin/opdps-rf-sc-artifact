#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from opdps_rf_sc.config import load_yaml
from opdps_rf_sc.models import build_separator
from opdps_rf_sc.train_utils import autocast_context, configure_torch, count_parameters, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/proposed.yaml"))
    ap.add_argument("--stage", choices=["coarse", "diffusion"], default="coarse")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--length", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    configure_torch()
    device = resolve_device(args.device)
    if device.type != "cuda":
        print("WARNING: CUDA is not available; this is only a shape preflight.")
    if args.stage == "coarse":
        mcfg = cfg["model"]; tcfg = cfg["train"]; in_ch = 2
    else:
        mcfg = cfg["diffusion"]["model"]; tcfg = cfg["diffusion"]; in_ch = 6
    bs = args.batch_size or int(tcfg["batch_size"])
    length = args.length or int(cfg["train"].get("length", 40960))
    model = build_separator(mcfg, in_channels=in_ch, out_channels=2, time_conditioned=args.stage == "diffusion").to(device)
    x = torch.randn(bs, in_ch, length, device=device)
    cls = torch.randint(0, 2, (bs,), device=device)
    t = torch.randint(0, 1000, (bs,), device=device).float() if args.stage == "diffusion" else None
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with autocast_context(device, device.type == "cuda"):
        y = model(x, cls, t=t)
        loss = y.square().mean()
    loss.backward()
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"stage={args.stage} params={count_parameters(model):,} batch={bs} length={length} elapsed={elapsed:.3f}s")
    if device.type == "cuda":
        print(f"peak_allocated_GB={torch.cuda.max_memory_allocated()/1e9:.3f}")
        print(f"gpu={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
