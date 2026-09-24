from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999).float()


@dataclass
class DiffusionSchedule:
    timesteps: int = 1000

    def __post_init__(self):
        self.betas = cosine_beta_schedule(self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bar[t].reshape(-1, 1, 1).to(dtype=x0.dtype)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    def predict_x0(self, xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bar[t].reshape(-1, 1, 1).to(dtype=xt.dtype)
        return (xt - (1.0 - a).sqrt() * eps) / a.sqrt().clamp_min(1e-6)


def _model_eps(model, x, mixture, coarse, class_id, t):
    inp = torch.cat([x, mixture, coarse], dim=1)
    return model(inp, class_id, t=t.float())


@torch.no_grad()
def sample_ddim(
    model,
    mixture: torch.Tensor,
    coarse: torch.Tensor,
    class_id: torch.Tensor,
    schedule: DiffusionSchedule,
    steps: int = 30,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Deterministic DDIM (eta=0) sampler for the normalized residual."""
    device = mixture.device
    b, _, length = mixture.shape
    x = torch.randn((b, 2, length), device=device, dtype=mixture.dtype, generator=generator)
    ts = torch.linspace(schedule.timesteps - 1, 0, steps, device=device).round().long()
    # Remove duplicates that can appear for pathological step counts.
    ts = torch.unique_consecutive(ts)
    for j, t_scalar in enumerate(ts):
        t = torch.full((b,), int(t_scalar.item()), device=device, dtype=torch.long)
        eps = _model_eps(model, x, mixture, coarse, class_id, t)
        a_t = schedule.alpha_bar[t_scalar].to(device=device, dtype=x.dtype)
        x0 = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t).clamp_min(1e-6)
        if j == len(ts) - 1:
            x = x0
            break
        t_next = ts[j + 1]
        a_next = schedule.alpha_bar[t_next].to(device=device, dtype=x.dtype)
        x = torch.sqrt(a_next) * x0 + torch.sqrt(1 - a_next) * eps
    return x


@torch.no_grad()
def sample_dpmsolver_pp(
    model,
    mixture: torch.Tensor,
    coarse: torch.Tensor,
    class_id: torch.Tensor,
    schedule: DiffusionSchedule,
    steps: int = 20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """DPM-Solver++(2M) via diffusers, using the exact betas used for training."""
    try:
        from diffusers import DPMSolverMultistepScheduler
    except Exception as exc:
        raise RuntimeError(
            "DPM-Solver++ requires the `diffusers` package. Install with `pip install -r requirements.txt`, "
            "or select --sampler ddim."
        ) from exc

    device = mixture.device
    b, _, length = mixture.shape
    x = torch.randn((b, 2, length), device=device, dtype=mixture.dtype, generator=generator)

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=schedule.timesteps,
        trained_betas=schedule.betas.detach().cpu().numpy(),
        prediction_type="epsilon",
        algorithm_type="dpmsolver++",
        solver_order=2,
        lower_order_final=True,
        lambda_min_clipped=-5.1,
        )
    scheduler.set_timesteps(steps, device=device)
    for t_scalar in scheduler.timesteps:
        model_in = scheduler.scale_model_input(x, t_scalar)
        t = torch.full((b,), int(t_scalar.item()), device=device, dtype=torch.long)
        eps = _model_eps(model, model_in, mixture, coarse, class_id, t)
        # diffusers scheduler signatures have changed slightly across releases.
        # DPM-Solver++ itself is deterministic for a fixed initial x, so the
        # generator is only needed by versions that explicitly accept it.
        try:
            x = scheduler.step(eps, t_scalar, x, generator=generator).prev_sample
        except TypeError:
            x = scheduler.step(eps, t_scalar, x).prev_sample
    return x


@torch.no_grad()
def sample_residual_ensemble(
    model,
    mixture: torch.Tensor,
    coarse: torch.Tensor,
    class_id: torch.Tensor,
    schedule: DiffusionSchedule,
    scales: torch.Tensor,
    num_samples: int = 4,
    steps: int = 20,
    sampler: str = "dpm",
    blend: float = 1.0,
    seed: int = 0,
) -> torch.Tensor:
    """Return coarse + posterior-mean residual. Averaging samples targets the MSE-optimal mean."""
    b = mixture.shape[0]
    preds = []
    for k in range(num_samples):
        gen = torch.Generator(device=mixture.device)
        gen.manual_seed(seed + k)
        if sampler in {"dpm", "dpm_solver", "dpm++"}:
            r_norm = sample_dpmsolver_pp(model, mixture, coarse, class_id, schedule, steps=steps, generator=gen)
        elif sampler == "ddim":
            r_norm = sample_ddim(model, mixture, coarse, class_id, schedule, steps=steps, generator=gen)
        else:
            raise ValueError(f"Unknown sampler {sampler}")
        scale = scales[class_id].reshape(b, 1, 1).to(dtype=coarse.dtype, device=coarse.device)
        preds.append(coarse + float(blend) * scale * r_norm)
    return torch.stack(preds, dim=0).mean(dim=0)
