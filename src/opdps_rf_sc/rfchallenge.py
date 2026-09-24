from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .constants import OFFICIAL_MIXTURE_LENGTH


@contextmanager
def pushd(path: str | Path):
    old = Path.cwd()
    os.chdir(Path(path).expanduser().resolve())
    try:
        yield
    finally:
        os.chdir(old)


def validate_rf_root(rf_root: str | Path) -> Path:
    root = Path(rf_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"RFChallenge root does not exist: {root}")
    if not (root / "dataset").exists():
        raise FileNotFoundError(
            f"Expected {root / 'dataset'}. Point --rf-root to the inner "
            "rfchallenge_singlechannel_starter-main directory that contains dataset/."
        )
    return root


def import_rfcutils(rf_root: str | Path) -> Any:
    root = validate_rf_root(rf_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    with pushd(root):
        try:
            mod = importlib.import_module("rfcutils")
        except Exception as exc:
            raise RuntimeError(
                "Could not import the official starter's rfcutils package. "
                "Make sure --rf-root points to the extracted GitHub starter root and "
                "that its RF-env/dependencies are installed."
            ) from exc
    return mod


class RFCAdapter:
    """Thin wrapper around the official starter APIs used in QuickStart.ipynb."""

    def __init__(self, rf_root: str | Path):
        self.root = validate_rf_root(rf_root)
        self.rfcutils = import_rfcutils(self.root)

    def load_sample(self, idx: int, dataset_type: str, signal_type: str):
        with pushd(self.root):
            return self.rfcutils.load_dataset_sample(idx, dataset_type, signal_type)

    def load_components(self, idx: int, dataset_type: str, interference_type: str):
        with pushd(self.root):
            return self.rfcutils.load_dataset_sample_components(idx, dataset_type, interference_type)

    def create_sep_mixture(self, interference_type: str, target_sinr_db: float, dataset_type: str = "train"):
        with pushd(self.root):
            return self.rfcutils.create_sep_mixture(
                interference_type,
                target_sinr_db=float(target_sinr_db),
                dataset_type=dataset_type,
            )


def validate_official_separation_example(adapter: RFCAdapter, interference_type: str = "EMISignal1", idx: int = 401):
    mix, meta = adapter.load_sample(idx, "sep_val", interference_type)
    soi, meta_soi, intr, meta_intr = adapter.load_components(idx, "sep_val", interference_type)
    mix = np.asarray(mix)
    soi = np.asarray(soi)
    intr = np.asarray(intr)
    if mix.shape[-1] != OFFICIAL_MIXTURE_LENGTH:
        raise ValueError(f"Expected validation mixture length {OFFICIAL_MIXTURE_LENGTH}, got {mix.shape}")
    if not np.allclose(mix, soi + intr, rtol=2e-5, atol=2e-5):
        maxerr = float(np.max(np.abs(mix - (soi + intr))))
        raise ValueError(f"Official mixture != component1 + component2; max error={maxerr:g}")
    return mix, soi, intr, meta, meta_soi, meta_intr
