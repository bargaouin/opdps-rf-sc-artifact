from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import OperatorBlock1D, ResBlock1D, SinusoidalEmbedding


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, time_conditioned: bool):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.time_conditioned = bool(time_conditioned)
        if self.time_conditioned:
            self.time_pos = SinusoidalEmbedding(emb_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                nn.SiLU(),
                nn.Linear(emb_dim * 2, emb_dim),
            )
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, class_id: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.class_emb(class_id)
        if self.time_conditioned:
            if t is None:
                raise ValueError("This model requires a diffusion timestep")
            if t.ndim == 0:
                t = t.expand(class_id.shape[0])
            elif t.numel() == 1:
                t = t.reshape(1).expand(class_id.shape[0])
            emb = emb + self.time_mlp(self.time_pos(t))
        return self.out(emb)


class HybridOperatorUNet1D(nn.Module):
    """Length-agnostic 1-D U-Net with global Fourier-operator bottleneck blocks."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_classes: int = 2,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 6),
        emb_dim: int = 256,
        operator_blocks: int = 6,
        max_modes: int = 256,
        spectral_rank: int = 32,
        dropout: float = 0.0,
        time_conditioned: bool = False,
        residual_to_input: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.time_conditioned = bool(time_conditioned)
        self.residual_to_input = bool(residual_to_input)
        widths = [base_channels * int(m) for m in channel_mults]
        self.cond = ConditionalEmbedding(num_classes, emb_dim, time_conditioned=time_conditioned)
        self.stem = nn.Conv1d(in_channels, widths[0], 7, padding=3)

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.enc_blocks.append(ResBlock1D(widths[i], widths[i], emb_dim, dropout=dropout))
            self.down.append(nn.Conv1d(widths[i], widths[i + 1], 4, stride=2, padding=1))

        self.mid_in = ResBlock1D(widths[-1], widths[-1], emb_dim, dropout=dropout)
        self.operator = nn.ModuleList(
            [OperatorBlock1D(widths[-1], emb_dim, max_modes=max_modes, dropout=dropout, spectral_rank=spectral_rank) for _ in range(operator_blocks)]
        )
        self.mid_out = ResBlock1D(widths[-1], widths[-1], emb_dim, dropout=dropout)

        self.up_proj = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        current = widths[-1]
        for skip_w in reversed(widths[:-1]):
            self.up_proj.append(nn.Conv1d(current, skip_w, 1))
            self.dec_blocks.append(ResBlock1D(2 * skip_w, skip_w, emb_dim, dropout=dropout))
            current = skip_w
        self.out_norm = nn.GroupNorm(8 if widths[0] % 8 == 0 else 1, widths[0])
        self.out = nn.Conv1d(widths[0], out_channels, 7, padding=3)

    def forward(self, x: torch.Tensor, class_id: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        input_x = x
        emb = self.cond(class_id, t=t)
        h = self.stem(x)
        skips = []
        for block, down in zip(self.enc_blocks, self.down):
            h = block(h, emb)
            skips.append(h)
            h = down(h)

        h = self.mid_in(h, emb)
        for block in self.operator:
            h = block(h, emb)
        h = self.mid_out(h, emb)

        for proj, block, skip in zip(self.up_proj, self.dec_blocks, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = proj(h)
            h = block(torch.cat([h, skip], dim=1), emb)
        out = self.out(F.silu(self.out_norm(h)))
        if self.residual_to_input and input_x.shape[1] >= self.out_channels:
            out = out + input_x[:, : self.out_channels]
        return out


class FNOSeparator1D(nn.Module):
    """Direct Fourier neural operator baseline without a U-Net hierarchy."""

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        num_classes=2,
        width=48,
        emb_dim=192,
        operator_blocks=8,
        max_modes=192,
        spectral_rank=24,
        dropout=0.0,
        residual_to_input=True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.residual_to_input = residual_to_input
        self.cond = ConditionalEmbedding(num_classes, emb_dim, time_conditioned=False)
        self.stem = nn.Conv1d(in_channels, width, 1)
        self.blocks = nn.ModuleList(
            [OperatorBlock1D(width, emb_dim, max_modes=max_modes, dropout=dropout, spectral_rank=spectral_rank) for _ in range(operator_blocks)]
        )
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, out_channels, 1))

    def forward(self, x, class_id, t=None):
        emb = self.cond(class_id)
        h = self.stem(x)
        for block in self.blocks:
            h = block(h, emb)
        out = self.head(h)
        if self.residual_to_input:
            out = out + x[:, : self.out_channels]
        return out


def build_separator(model_cfg: dict, *, in_channels: int = 2, out_channels: int = 2, time_conditioned: bool = False):
    kind = model_cfg.get("kind", "hybrid_operator")
    common = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        num_classes=int(model_cfg.get("num_classes", 2)),
    )
    if kind == "hybrid_operator":
        return HybridOperatorUNet1D(
            **common,
            base_channels=int(model_cfg.get("base_channels", 64)),
            channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4, 6])),
            emb_dim=int(model_cfg.get("emb_dim", 256)),
            operator_blocks=int(model_cfg.get("operator_blocks", 6)),
            max_modes=int(model_cfg.get("max_modes", 256)),
            spectral_rank=int(model_cfg.get("spectral_rank", 32)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            time_conditioned=time_conditioned,
            residual_to_input=bool(model_cfg.get("residual_to_input", not time_conditioned)),
        )
    if kind == "waveunet":
        return HybridOperatorUNet1D(
            **common,
            base_channels=int(model_cfg.get("base_channels", 64)),
            channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4, 6])),
            emb_dim=int(model_cfg.get("emb_dim", 256)),
            operator_blocks=0,
            max_modes=1,
            spectral_rank=1,
            dropout=float(model_cfg.get("dropout", 0.0)),
            time_conditioned=time_conditioned,
            residual_to_input=bool(model_cfg.get("residual_to_input", not time_conditioned)),
        )
    if kind == "fno":
        if time_conditioned:
            raise ValueError("FNOSeparator1D baseline is direct-only; use hybrid_operator or waveunet for diffusion")
        return FNOSeparator1D(
            **common,
            width=int(model_cfg.get("width", 48)),
            emb_dim=int(model_cfg.get("emb_dim", 192)),
            operator_blocks=int(model_cfg.get("operator_blocks", 8)),
            max_modes=int(model_cfg.get("max_modes", 192)),
            spectral_rank=int(model_cfg.get("spectral_rank", 24)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            residual_to_input=bool(model_cfg.get("residual_to_input", True)),
        )
    raise ValueError(f"Unknown model kind: {kind}")
