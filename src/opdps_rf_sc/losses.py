from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .complex_utils import twoch_to_torch_complex


def complex_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Mean |e|^2, which is exactly the natural complex MSE used by the challenge.
    return (pred - target).square().sum(dim=1).mean()


def complex_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    e = pred - target
    return torch.sqrt(e.square().sum(dim=1) + 1e-12).mean()


def spectral_logmag_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    p = twoch_to_torch_complex(pred)
    t = twoch_to_torch_complex(target)
    pf = torch.fft.fft(p.float() if not torch.is_complex(p) else p, dim=-1, norm="ortho")
    tf = torch.fft.fft(t.float() if not torch.is_complex(t) else t, dim=-1, norm="ortho")
    return F.l1_loss(torch.log(torch.abs(pf) + eps), torch.log(torch.abs(tf) + eps))


def separation_loss(pred: torch.Tensor, target: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    mse = complex_mse(pred, target)
    l1_w = float(cfg.get("l1_weight", 0.05))
    spec_w = float(cfg.get("spectral_weight", 0.02))
    l1 = complex_l1(pred, target) if l1_w > 0 else pred.new_zeros(())
    spec = spectral_logmag_loss(pred, target) if spec_w > 0 else pred.new_zeros(())
    loss = mse + l1_w * l1 + spec_w * spec
    return loss, {
        "loss": float(loss.detach()),
        "mse": float(mse.detach()),
        "l1": float(l1.detach()),
        "spectral": float(spec.detach()),
    }
