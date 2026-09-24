#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opdps_rf_sc.model_io import load_coarse_from_checkpoint


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RUN_ROOT = Path(os.environ["RUN_ROOT"])
CACHE_DIR = Path(os.environ["CACHE_DIR"])

OUTDIR = RUN_ROOT / "paper_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "Direct FNO": RUN_ROOT / "baseline_fno_v100/coarse/best.pt",
    "LTS": RUN_ROOT / "baseline_wavenet_v100/coarse/best.pt",
    "CM-LTS": RUN_ROOT / "baseline_wavenet_parammatch_v100/coarse/best.pt",
    "FETS": RUN_ROOT / "proposed_v100/coarse/stage1_final_best.pt",
}

# Prefer the official fixed 2,200-mixture evaluation.
metric_candidates = [
    RUN_ROOT / "evals/final_coarse_v100/metrics.csv",
    RUN_ROOT / "evals/proposed_90k_v100/metrics.csv",
]

METRICS = next((p for p in metric_candidates if p.exists()), None)
if METRICS is None:
    raise FileNotFoundError(
        "Could not find FETS metrics.csv in:\n"
        + "\n".join(str(p) for p in metric_candidates)
    )


# ---------------------------------------------------------------------
# Load fixed RFChallenge evaluation cache
# ---------------------------------------------------------------------

mixture = np.load(CACHE_DIR / "sep_val_mixture.npy", mmap_mode="r")
target = np.load(CACHE_DIR / "sep_val_target.npy", mmap_mode="r")
class_id = np.load(CACHE_DIR / "sep_val_class_id.npy", mmap_mode="r")

print("cache:")
print(" mixture:", mixture.shape, mixture.dtype)
print(" target :", target.shape, target.dtype)
print(" class  :", class_id.shape, class_id.dtype)

CLASS_NAME = {
    0: "EMISignal1",
    1: "CommSignal3",
}
NAME_TO_CLASS = {v: k for k, v in CLASS_NAME.items()}

class_rows = {
    cid: np.flatnonzero(np.asarray(class_id) == cid)
    for cid in CLASS_NAME
}


# ---------------------------------------------------------------------
# Pick representative samples
# ---------------------------------------------------------------------

df = pd.read_csv(METRICS)

print("\nmetrics:", METRICS)
print("columns:", list(df.columns))

required = {"interference", "sinr_db", "nmse_soi_db"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"metrics.csv missing required columns: {missing}")


def metric_row_to_cache_row(r: pd.Series) -> int:
    """
    Convert metrics.csv row to the corresponding cache row.

    Handles:
      cache_row / row / cache_idx
      per-class idx
      global idx
      or falls back to dataframe row order.
    """

    for key in ("cache_row", "cache_idx", "row"):
        if key in r.index and pd.notna(r[key]):
            return int(r[key])

    if "idx" in r.index and pd.notna(r["idx"]):
        idx = int(r["idx"])
        name = str(r["interference"])
        cid = NAME_TO_CLASS[name]

        # In the official evaluator idx may be within-class.
        if 0 <= idx < len(class_rows[cid]):
            return int(class_rows[cid][idx])

        # Otherwise treat it as a global row.
        if 0 <= idx < len(mixture):
            return idx

    # Last-resort assumption: metrics row order == cache row order.
    return int(r.name)


def choose_representative(interference: str, sinr_db: float):
    sub = df[
        (df["interference"] == interference)
        & np.isclose(df["sinr_db"].astype(float), sinr_db)
    ].copy()

    if len(sub) == 0:
        raise RuntimeError(
            f"No samples for {interference}, SINR={sinr_db}"
        )

    median_nmse = sub["nmse_soi_db"].median()
    chosen_index = (sub["nmse_soi_db"] - median_nmse).abs().idxmin()
    r = sub.loc[chosen_index]

    cache_row = metric_row_to_cache_row(r)

    return {
        "interference": interference,
        "sinr_db": float(sinr_db),
        "cache_row": cache_row,
        "fets_eval_nmse_db": float(r["nmse_soi_db"]),
    }


CASES = [
    choose_representative("EMISignal1", 3.0),
    choose_representative("CommSignal3", -4.5),
    choose_representative("CommSignal3", -12.0),
]

print("\nSelected representative cases:")
for c in CASES:
    print(c)


# ---------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\ndevice:", device)

models = {}

for name, path in MODEL_PATHS.items():
    if not path.exists():
        raise FileNotFoundError(f"{name}: checkpoint missing: {path}")

    model, ckpt = load_coarse_from_checkpoint(
        path,
        device,
        use_ema=True,
    )

    model.eval()
    models[name] = model

    nparams = sum(p.numel() for p in model.parameters())
    print(
        f"{name:12s}: "
        f"step={ckpt.get('step')} "
        f"params={nparams:,}"
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def complex_to_2ch(z: np.ndarray) -> torch.Tensor:
    x = np.stack([z.real, z.imag], axis=0).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0)


def output_to_complex(x: torch.Tensor) -> np.ndarray:
    a = x.detach().float().cpu().numpy()[0]
    return a[0] + 1j * a[1]


def run_model(model, z: np.ndarray, cid: int) -> np.ndarray:
    x = complex_to_2ch(z).to(device)
    c = torch.tensor([cid], dtype=torch.long, device=device)

    with torch.no_grad():
        try:
            p = model(x, c)
        except TypeError:
            p = model(x, c, t=None)

    return output_to_complex(p)


def nmse_db(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.sum(np.abs(pred - truth) ** 2)
    den = np.sum(np.abs(truth) ** 2) + 1e-12
    return float(10.0 * np.log10(num / den))


def compute_stft(z: np.ndarray):
    f, t, Z = stft(
        z,
        fs=1.0,
        window="hann",
        nperseg=512,
        noverlap=384,
        nfft=1024,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    f = np.fft.fftshift(f)
    Z = np.fft.fftshift(Z, axes=0)

    return f, t, Z


def display_spectrogram(
    ax,
    z,
    reference,
    amp_limit,
    label=None,
    nmse=None,
):
    f, tt, Z = compute_stft(z)
    _, _, Zref = compute_stft(reference)

    ref_peak = np.max(np.abs(Zref)) + 1e-12

    db = 20.0 * np.log10(
        np.abs(Z) / ref_peak + 1e-10
    )

    im = ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=[0, len(z), -0.5, 0.5],
        vmin=-60,
        vmax=6,
        cmap="magma",
        interpolation="nearest",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    if label is not None:
        ax.set_title(label, fontsize=10, pad=4)

    if nmse is not None:
        ax.text(
            0.97,
            0.95,
            f"{nmse:.2f} dB",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="black",
                alpha=0.60,
                linewidth=0,
            ),
        )

    # Small time-domain I/Q inset.
    iax = inset_axes(
        ax,
        width="94%",
        height="25%",
        loc="lower center",
        borderpad=0.35,
    )

    seglen = min(1024, len(z))
    start = max(0, len(z) // 2 - seglen // 2)
    zz = z[start:start + seglen]
    xx = np.arange(seglen)

    iax.plot(xx, zz.real, linewidth=0.45, color="white", alpha=0.95)
    iax.plot(xx, zz.imag, linewidth=0.45, color="0.70", alpha=0.95)

    iax.set_xlim(0, seglen - 1)
    iax.set_ylim(-amp_limit, amp_limit)
    iax.set_xticks([])
    iax.set_yticks([])
    iax.set_facecolor((0, 0, 0, 0.38))

    for spine in iax.spines.values():
        spine.set_visible(False)

    return im


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

all_rows = []
plot_rows = []

for case in CASES:

    row = case["cache_row"]
    mix = np.asarray(mixture[row])
    gt = np.asarray(target[row])
    cid = int(class_id[row])

    predictions = {}

    for name, model in models.items():
        pred = run_model(model, mix, cid)
        predictions[name] = pred

    row_signals = {
        "Observed mixture": mix,
        "Ground truth": gt,
        **predictions,
    }

    # Same amplitude range across every panel in this row.
    amp_limit = max(
        max(
            np.max(np.abs(z.real)),
            np.max(np.abs(z.imag)),
        )
        for z in row_signals.values()
    )
    amp_limit *= 1.05

    plot_rows.append(
        {
            **case,
            "mixture": mix,
            "target": gt,
            "predictions": predictions,
            "amp_limit": amp_limit,
        }
    )

    record = {
        "interference": case["interference"],
        "sinr_db": case["sinr_db"],
        "cache_row": row,
    }

    for name, pred in predictions.items():
        record[f"{name}_nmse_db"] = nmse_db(pred, gt)

    all_rows.append(record)


manifest = pd.DataFrame(all_rows)
manifest_path = OUTDIR / "qualitative_selected_samples.csv"
manifest.to_csv(manifest_path, index=False)

print("\nQualitative sample metrics:")
print(manifest.to_string(index=False))


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

column_specs = [
    ("Observed mixture", None),
    ("Ground truth", None),
    ("Direct FNO", "Direct FNO"),
    ("LTS\n(8.94M)", "LTS"),
    ("CM-LTS\n(15.00M)", "CM-LTS"),
    ("FETS\n(14.84M)", "FETS"),
]

fig, axes = plt.subplots(
    nrows=len(plot_rows),
    ncols=len(column_specs),
    figsize=(14.8, 7.2),
    constrained_layout=False,
)

plt.subplots_adjust(
    left=0.105,
    right=0.965,
    top=0.91,
    bottom=0.10,
    wspace=0.08,
    hspace=0.18,
)

last_im = None

for i, case in enumerate(plot_rows):

    gt = case["target"]

    row_label = (
        f"{case['interference']}\n"
        f"SINR = {case['sinr_db']:g} dB"
    )

    for j, (display_name, model_key) in enumerate(column_specs):

        ax = axes[i, j]

        if display_name == "Observed mixture":
            z = case["mixture"]
            score = None

        elif display_name == "Ground truth":
            z = gt
            score = None

        else:
            z = case["predictions"][model_key]
            score = nmse_db(z, gt)

        title = display_name if i == 0 else None

        last_im = display_spectrogram(
            ax,
            z,
            reference=gt,
            amp_limit=case["amp_limit"],
            label=title,
            nmse=score,
        )

        if j == 0:
            ax.set_ylabel(
                row_label,
                fontsize=10,
                rotation=0,
                labelpad=49,
                va="center",
            )

# Add spectral-axis meaning only once.
fig.text(
    0.018,
    0.50,
    "Normalized frequency",
    rotation=90,
    va="center",
    fontsize=10,
)

fig.text(
    0.535,
    0.035,
    "Time",
    ha="center",
    fontsize=10,
)

cbar_ax = fig.add_axes([0.972, 0.16, 0.012, 0.67])

cb = fig.colorbar(
    last_im,
    cax=cbar_ax,
)

cb.set_label(
    "Magnitude relative to ground-truth peak (dB)",
    fontsize=9,
)
cb.ax.tick_params(labelsize=8)

png_path = OUTDIR / "qualitative_all_models.png"
pdf_path = OUTDIR / "qualitative_all_models.pdf"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

print("\nSaved:")
print(" ", png_path)
print(" ", pdf_path)
print(" ", manifest_path)
