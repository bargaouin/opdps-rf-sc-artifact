from __future__ import annotations

import torch

from .checkpoint import load_checkpoint
from .models import build_separator


def load_coarse_from_checkpoint(path, device, use_ema=True):
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = ckpt["config"]
    model = build_separator(cfg["model"], in_channels=2, out_channels=2, time_conditioned=False)
    state = ckpt.get("ema") if use_ema and ckpt.get("ema") is not None else ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt


def load_diffusion_from_checkpoint(path, device, use_ema=True):
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = ckpt["config"]
    dcfg = cfg["diffusion"]
    model = build_separator(dcfg["model"], in_channels=6, out_channels=2, time_conditioned=True)
    state = ckpt.get("ema") if use_ema and ckpt.get("ema") is not None else ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt
