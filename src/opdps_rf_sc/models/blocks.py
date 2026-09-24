from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int, preferred: int = 8) -> int:
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        half = self.dim // 2
        freq = torch.exp(
            torch.linspace(0, -math.log(10000.0), half, device=t.device, dtype=torch.float32)
        )
        args = t.float().reshape(-1, 1) * freq.reshape(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb)).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale[..., None]) + shift[..., None]
        h = self.conv2(self.dropout(F.silu(h)))
        return (h + self.skip(x)) / math.sqrt(2.0)


class SpectralConv1d(nn.Module):
    """Factorized complex Fourier operator.

    A dense [out,in,modes] complex tensor becomes enormous at RF widths.  We use a
    low-rank channel factorization with mode-dependent complex gains, preserving a
    global Fourier operator while keeping the H100 experiment practical.
    """

    def __init__(self, in_ch: int, out_ch: int, max_modes: int, rank: int = 32):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.max_modes = int(max_modes)
        self.rank = min(int(rank), in_ch, out_ch)
        scale = 1.0 / math.sqrt(max(1, in_ch))
        self.in_factor = nn.Parameter(scale * torch.randn(self.rank, in_ch))
        self.out_factor = nn.Parameter(scale * torch.randn(out_ch, self.rank))
        self.mode_real = nn.Parameter(0.02 * torch.randn(self.rank, self.max_modes))
        self.mode_imag = nn.Parameter(0.02 * torch.randn(self.rank, self.max_modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x.float(), dim=-1)
        nfreq = x_ft.shape[-1]
        modes = min(self.max_modes, nfreq)
        # Channel projection -> mode-wise complex gain -> channel expansion.
        z = torch.einsum("ri,bim->brm", self.in_factor.to(x_ft.dtype), x_ft[..., :modes])
        gain = torch.complex(self.mode_real[:, :modes], self.mode_imag[:, :modes])
        z = z * gain[None, ...]
        low = torch.einsum("or,brm->bom", self.out_factor.to(z.dtype), z)
        out_ft = torch.zeros(x.shape[0], self.out_ch, nfreq, device=x.device, dtype=torch.complex64)
        out_ft[..., :modes] = low
        out = torch.fft.irfft(out_ft, n=n, dim=-1)
        return out.to(dtype=x.dtype)


class OperatorBlock1D(nn.Module):
    def __init__(self, channels: int, emb_dim: int, max_modes: int, dropout: float = 0.0, spectral_rank: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.spec = SpectralConv1d(channels, channels, max_modes=max_modes, rank=spectral_rank)
        self.local = nn.Conv1d(channels, channels, 9, padding=4, groups=channels)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.emb = nn.Linear(emb_dim, 2 * channels)
        self.ff_norm = nn.GroupNorm(_groups(channels), channels)
        self.ff = nn.Sequential(
            nn.Conv1d(channels, 2 * channels, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(2 * channels, channels, 1),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.spec(h) + self.local(h)
        scale, shift = self.emb(F.silu(emb)).chunk(2, dim=-1)
        h = h * (1.0 + scale[..., None]) + shift[..., None]
        x = (x + self.mix(F.gelu(h))) / math.sqrt(2.0)
        return (x + self.ff(self.ff_norm(x))) / math.sqrt(2.0)
