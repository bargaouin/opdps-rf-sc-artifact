from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def autocast_context(device: torch.device, enabled=True):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


def cosine_warmup_lambda(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.05):
    if step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def make_lr_scheduler(optimizer, total_steps: int, warmup_steps: int, min_ratio: float = 0.05):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_warmup_lambda(s, total_steps, warmup_steps, min_ratio),
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def atomic_json(path: str | Path, obj):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)
