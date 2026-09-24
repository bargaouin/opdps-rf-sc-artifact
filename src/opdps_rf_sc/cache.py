from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .constants import OFFICIAL_TRAIN_FRAME_COUNTS, SIGNAL_TYPES
from .rfchallenge import RFCAdapter


def build_frame_cache(
    rf_root: str | Path,
    cache_dir: str | Path,
    counts: Mapping[str, int] | None = None,
    holdout_last: int = 50,
    overwrite: bool = False,
) -> dict:
    """Convert official train_frame SigMF recordings to memory-mappable complex64 arrays.

    The challenge description states that sep_val is generated from the last 50 frames of
    each training signal class. By default those frames are marked as holdout and are never
    sampled by the training mixture generator.
    """
    counts = dict(counts or OFFICIAL_TRAIN_FRAME_COUNTS)
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text())

    adapter = RFCAdapter(rf_root)
    manifest = {
        "rf_root": str(Path(rf_root).expanduser().resolve()),
        "holdout_last": int(holdout_last),
        "signals": {},
    }

    for signal_type in SIGNAL_TYPES:
        n = int(counts[signal_type])
        first, _ = adapter.load_sample(0, "train_frame", signal_type)
        first = np.asarray(first, dtype=np.complex64).reshape(-1)
        frame_len = first.size
        out_path = cache_dir / f"{signal_type}.npy"
        arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.complex64, shape=(n, frame_len))
        arr[0] = first
        for i in range(1, n):
            x, _ = adapter.load_sample(i, "train_frame", signal_type)
            x = np.asarray(x, dtype=np.complex64).reshape(-1)
            if x.size != frame_len:
                raise ValueError(
                    f"Frame length changed inside {signal_type}: frame0={frame_len}, frame{i}={x.size}"
                )
            arr[i] = x
        arr.flush()
        train_count = max(0, n - holdout_last)
        if train_count == 0:
            raise ValueError(f"holdout_last={holdout_last} leaves no training frames for {signal_type}")
        manifest["signals"][signal_type] = {
            "path": out_path.name,
            "count": n,
            "frame_len": frame_len,
            "train_indices": [0, train_count],
            "holdout_indices": [train_count, n],
        }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


class FrameCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.arrays = {
            name: np.load(self.cache_dir / info["path"], mmap_mode="r")
            for name, info in self.manifest["signals"].items()
        }

    def bounds(self, signal_type: str, split: str = "train") -> tuple[int, int]:
        key = "train_indices" if split == "train" else "holdout_indices"
        lo, hi = self.manifest["signals"][signal_type][key]
        return int(lo), int(hi)

    def sample_window(
        self,
        signal_type: str,
        length: int,
        rng: np.random.Generator,
        split: str = "train",
    ) -> np.ndarray:
        arr = self.arrays[signal_type]
        lo, hi = self.bounds(signal_type, split)
        if hi <= lo:
            raise ValueError(f"Empty {split} split for {signal_type}")
        frame_idx = int(rng.integers(lo, hi))
        frame = arr[frame_idx]
        if length > frame.shape[-1]:
            raise ValueError(f"Requested {length} samples from {signal_type} frame of length {frame.shape[-1]}")
        start = int(rng.integers(0, frame.shape[-1] - length + 1))
        return np.asarray(frame[start : start + length], dtype=np.complex64)


def build_sep_val_cache(
    rf_root: str | Path,
    cache_dir: str | Path,
    examples_per_interference: int = 1100,
    overwrite: bool = False,
) -> dict:
    """Cache official sep_val mixture/CommSignal2 target arrays for fast GPU evaluation.

    This step uses the official rfcutils once; afterwards evaluation does not need to
    parse SigMF files or import the legacy starter dependencies.
    """
    from .constants import OFFICIAL_MIXTURE_LENGTH, SEPARATION_INTERFERERS, INTERFERENCE_TO_ID

    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Build the train_frame cache first")
    manifest = json.loads(manifest_path.read_text())
    mix_path = cache_dir / "sep_val_mixture.npy"
    target_path = cache_dir / "sep_val_target.npy"
    class_path = cache_dir / "sep_val_class_id.npy"
    idx_path = cache_dir / "sep_val_idx.npy"
    total = len(SEPARATION_INTERFERERS) * int(examples_per_interference)
    if (
        not overwrite
        and mix_path.exists()
        and target_path.exists()
        and class_path.exists()
        and idx_path.exists()
        and "sep_val" in manifest
    ):
        return manifest

    adapter = RFCAdapter(rf_root)
    mix = np.lib.format.open_memmap(
        mix_path, mode="w+", dtype=np.complex64, shape=(total, OFFICIAL_MIXTURE_LENGTH)
    )
    target = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.complex64, shape=(total, OFFICIAL_MIXTURE_LENGTH)
    )
    class_id = np.lib.format.open_memmap(class_path, mode="w+", dtype=np.int64, shape=(total,))
    sample_idx = np.lib.format.open_memmap(idx_path, mode="w+", dtype=np.int64, shape=(total,))

    row = 0
    for intr in SEPARATION_INTERFERERS:
        for idx in range(examples_per_interference):
            y, _ = adapter.load_sample(idx, "sep_val", intr)
            s, _, b, _ = adapter.load_components(idx, "sep_val", intr)
            y = np.asarray(y, dtype=np.complex64).reshape(-1)
            s = np.asarray(s, dtype=np.complex64).reshape(-1)
            b = np.asarray(b, dtype=np.complex64).reshape(-1)
            if y.size != OFFICIAL_MIXTURE_LENGTH:
                raise ValueError(f"Unexpected sep_val length {y.size} for {intr}/{idx}")
            if not np.allclose(y, s + b, rtol=2e-5, atol=2e-5):
                raise ValueError(f"sep_val consistency check failed for {intr}/{idx}")
            mix[row] = y
            target[row] = s
            class_id[row] = INTERFERENCE_TO_ID[intr]
            sample_idx[row] = idx
            row += 1
        print(f"cached sep_val {intr}: {examples_per_interference} examples", flush=True)

    for arr in (mix, target, class_id, sample_idx):
        arr.flush()
    manifest["sep_val"] = {
        "mixture_path": mix_path.name,
        "target_path": target_path.name,
        "class_id_path": class_path.name,
        "idx_path": idx_path.name,
        "examples_per_interference": int(examples_per_interference),
        "total": int(total),
        "length": OFFICIAL_MIXTURE_LENGTH,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


class SepValCache:
    def __init__(self, cache_dir: str | Path):
        cache_dir = Path(cache_dir).expanduser().resolve()
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        if "sep_val" not in manifest:
            raise FileNotFoundError(
                "sep_val cache missing. Re-run scripts/prepare_cache.py without --skip-sep-val."
            )
        info = manifest["sep_val"]
        self.mixture = np.load(cache_dir / info["mixture_path"], mmap_mode="r")
        self.target = np.load(cache_dir / info["target_path"], mmap_mode="r")
        self.class_id = np.load(cache_dir / info["class_id_path"], mmap_mode="r")
        self.idx = np.load(cache_dir / info["idx_path"], mmap_mode="r")
        self.info = info
