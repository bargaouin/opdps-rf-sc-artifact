from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .cache import FrameCache
from .complex_utils import np_complex_to_2ch
from .constants import (
    INTERFERENCE_TO_ID,
    OFFICIAL_MIXTURE_LENGTH,
    OFFICIAL_SINR_LEVELS_DESC,
    SEPARATION_INTERFERERS,
)


def normalize_unit_power(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = float(np.mean(np.abs(x) ** 2))
    return (x / np.sqrt(max(p, eps))).astype(np.complex64, copy=False)


def scale_interference_for_sinr(signal: np.ndarray, interference: np.ndarray, sinr_db: float) -> np.ndarray:
    ps = float(np.mean(np.abs(signal) ** 2))
    pi = float(np.mean(np.abs(interference) ** 2))
    target_ratio = 10.0 ** (float(sinr_db) / 10.0)
    scale = np.sqrt(ps / (max(pi, 1e-12) * target_ratio))
    return (interference * scale).astype(np.complex64, copy=False)


@dataclass
class MixtureBatch:
    mixture: torch.Tensor
    target: torch.Tensor
    interference: torch.Tensor
    class_id: torch.Tensor
    sinr_db: torch.Tensor


class MixtureBatchGenerator:
    """Fast on-the-fly generator matching the official y=s+b construction.

    It samples raw frames from the cached clean train_frame recordings, crops a window,
    scales CommSignal2 to unit power, and scales the chosen interferer to the requested SINR.
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        length: int = OFFICIAL_MIXTURE_LENGTH,
        sinr_levels: Iterable[float] = OFFICIAL_SINR_LEVELS_DESC,
        continuous_sinr_prob: float = 0.20,
        continuous_sinr_range: tuple[float, float] = (-15.0, 5.0),
        low_sinr_bias: float = 0.35,
        common_phase_aug_prob: float = 0.5,
        seed: int = 1234,
    ):
        self.cache = FrameCache(cache_dir)
        self.split = split
        self.length = int(length)
        self.sinr_levels = np.asarray(tuple(sinr_levels), dtype=np.float32)
        self.continuous_sinr_prob = float(continuous_sinr_prob)
        self.continuous_sinr_range = tuple(map(float, continuous_sinr_range))
        self.low_sinr_bias = float(low_sinr_bias)
        self.common_phase_aug_prob = float(common_phase_aug_prob)
        self.rng = np.random.default_rng(seed)

    def _draw_sinr(self) -> float:
        if self.rng.random() < self.continuous_sinr_prob:
            lo, hi = self.continuous_sinr_range
            return float(self.rng.uniform(lo, hi))
        if self.rng.random() < self.low_sinr_bias:
            candidates = self.sinr_levels[self.sinr_levels <= -6.0]
            return float(self.rng.choice(candidates))
        return float(self.rng.choice(self.sinr_levels))

    def sample_numpy(self, batch_size: int):
        mixtures = np.empty((batch_size, 2, self.length), dtype=np.float32)
        targets = np.empty_like(mixtures)
        interfs = np.empty_like(mixtures)
        cls = np.empty((batch_size,), dtype=np.int64)
        sinrs = np.empty((batch_size,), dtype=np.float32)

        for b in range(batch_size):
            int_type = SEPARATION_INTERFERERS[int(self.rng.integers(0, len(SEPARATION_INTERFERERS)))]
            sinr = self._draw_sinr()
            s = self.cache.sample_window("CommSignal2", self.length, self.rng, self.split)
            i = self.cache.sample_window(int_type, self.length, self.rng, self.split)
            s = normalize_unit_power(s)
            i = scale_interference_for_sinr(s, i, sinr)
            if self.rng.random() < self.common_phase_aug_prob:
                phase = np.complex64(np.exp(1j * self.rng.uniform(-np.pi, np.pi)))
                s = (s * phase).astype(np.complex64, copy=False)
                i = (i * phase).astype(np.complex64, copy=False)
            y = (s + i).astype(np.complex64, copy=False)
            mixtures[b] = np_complex_to_2ch(y)
            targets[b] = np_complex_to_2ch(s)
            interfs[b] = np_complex_to_2ch(i)
            cls[b] = INTERFERENCE_TO_ID[int_type]
            sinrs[b] = sinr
        return mixtures, targets, interfs, cls, sinrs

    def batch(self, batch_size: int, device: torch.device | str) -> MixtureBatch:
        mix, target, intr, cls, sinr = self.sample_numpy(batch_size)
        return MixtureBatch(
            mixture=torch.from_numpy(mix).to(device=device, non_blocking=True),
            target=torch.from_numpy(target).to(device=device, non_blocking=True),
            interference=torch.from_numpy(intr).to(device=device, non_blocking=True),
            class_id=torch.from_numpy(cls).to(device=device, non_blocking=True),
            sinr_db=torch.from_numpy(sinr).to(device=device, non_blocking=True),
        )
