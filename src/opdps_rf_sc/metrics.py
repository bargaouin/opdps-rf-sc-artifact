from __future__ import annotations

import numpy as np
import torch


def mse_2ch_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().sum(dim=-2).mean(dim=-1)


def mse_complex_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(target)) ** 2))


def nmse_db_complex_np(pred: np.ndarray, target: np.ndarray, eps=1e-12) -> float:
    num = np.mean(np.abs(np.asarray(pred) - np.asarray(target)) ** 2)
    den = np.mean(np.abs(np.asarray(target)) ** 2)
    return float(10 * np.log10((num + eps) / (den + eps)))
