from __future__ import annotations

import numpy as np
import torch


def np_complex_to_2ch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.stack((x.real, x.imag), axis=-2).astype(np.float32, copy=False)


def torch_complex_to_2ch(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(x):
        raise TypeError("Expected a complex tensor")
    return torch.stack((x.real, x.imag), dim=-2)


def twoch_to_torch_complex(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2] != 2:
        raise ValueError(f"Expected channel dimension of 2, got {tuple(x.shape)}")
    return torch.complex(x[..., 0, :], x[..., 1, :])


def twoch_to_np_complex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[-2] != 2:
        raise ValueError(f"Expected channel dimension of 2, got {x.shape}")
    return x[..., 0, :] + 1j * x[..., 1, :]


def complex_power_np(x: np.ndarray, axis=-1) -> np.ndarray:
    return np.mean(np.abs(x) ** 2, axis=axis)


def complex_power_2ch_torch(x: torch.Tensor, dim=-1) -> torch.Tensor:
    if x.shape[-2] != 2:
        raise ValueError("Expected I/Q channels")
    return (x.square().sum(dim=-2)).mean(dim=dim)


def measured_sinr_db_np(signal: np.ndarray, interference: np.ndarray) -> float:
    ps = float(np.mean(np.abs(signal) ** 2))
    pi = float(np.mean(np.abs(interference) ** 2))
    return 10.0 * np.log10(max(ps, 1e-30) / max(pi, 1e-30))
