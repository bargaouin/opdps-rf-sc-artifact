from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_cache(tmp_path: Path):
    rng = np.random.default_rng(0)
    signals = {}
    specs = {"EMISignal1": (8, 1024), "CommSignal2": (8, 1024), "CommSignal3": (8, 1024)}
    for name, (n, l) in specs.items():
        x = (rng.standard_normal((n, l)) + 1j * rng.standard_normal((n, l))).astype(np.complex64)
        # Add mild class-specific structure so the arrays are not identical white noise.
        if name == "CommSignal2":
            t = np.arange(l)[None, :]
            x += (0.5 * np.exp(1j * 2 * np.pi * 0.07 * t)).astype(np.complex64)
        elif name == "CommSignal3":
            t = np.arange(l)[None, :]
            x += (0.4 * np.exp(1j * 2 * np.pi * 0.19 * t)).astype(np.complex64)
        path = tmp_path / f"{name}.npy"
        np.save(path, x)
        signals[name] = {
            "path": path.name,
            "count": n,
            "frame_len": l,
            "train_indices": [0, 6],
            "holdout_indices": [6, 8],
        }
    manifest = {"rf_root": "synthetic", "holdout_last": 2, "signals": signals}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return tmp_path
