#!/usr/bin/env python

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opdps_rf_sc.model_io import load_coarse_from_checkpoint

RUN_ROOT = Path(os.environ["RUN_ROOT"])
CACHE_DIR = Path(os.environ["CACHE_DIR"])

OUT = RUN_ROOT / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# FILES
# ============================================================

FETS_METRICS = RUN_ROOT / "evals/final_coarse_v100/metrics.csv"

if not FETS_METRICS.exists():
    FETS_METRICS = RUN_ROOT / "evals/proposed_90k_v100/metrics.csv"

CKPTS = {
    "Direct FNO":
        RUN_ROOT / "baseline_fno_v100/coarse/best.pt",

    "CM-LTS":
        RUN_ROOT / "baseline_waveunet_parammatch_v100/coarse/best.pt",

    "FETS":
        RUN_ROOT / "proposed_v100/coarse/stage1_final_best.pt",
}

print("\n===== INPUT FILE CHECK =====")
print("FETS metrics:", FETS_METRICS)

if not FETS_METRICS.exists():
    raise FileNotFoundError(
        f"FETS metrics missing: {FETS_METRICS}"
    )

for name, path in CKPTS.items():
    print(f"{name:12s}: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"{name} checkpoint missing: {path}"
        )

cache_files = {
    "mixture": CACHE_DIR / "sep_val_mixture.npy",
    "target":  CACHE_DIR / "sep_val_target.npy",
    "class":   CACHE_DIR / "sep_val_class_id.npy",
}

for name, path in cache_files.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Cache file missing ({name}): {path}"
        )

print("All required files found.")

# ============================================================
# LOAD FIXED TEST SET
# ============================================================

mixture = np.load(
    cache_files["mixture"],
    mmap_mode="r"
)

target = np.load(
    cache_files["target"],
    mmap_mode="r"
)

classid = np.load(
    cache_files["class"],
    mmap_mode="r"
)

metrics = pd.read_csv(FETS_METRICS)

print("\nmetrics shape:", metrics.shape)
print("metrics columns:", metrics.columns.tolist())

assert len(metrics) == 2200
assert len(mixture) == 2200
assert len(target) == 2200

# ============================================================
# MAP METRIC ROW TO CACHE ROW
# ============================================================

CLASS_ID = {
    "EMISignal1": 0,
    "CommSignal3": 1,
}

class_rows = {
    cid: np.flatnonzero(
        np.asarray(classid) == cid
    )
    for cid in [0, 1]
}

def cache_row_from_metric(r):

    # Prefer explicit full-cache row if it exists.
    for col in [
        "cache_row",
        "cache_idx",
        "row"
    ]:
        if (
            col in metrics.columns
            and pd.notna(r[col])
        ):
            return int(r[col])

    # Current evaluation CSV uses idx within interference class.
    if (
        "idx" in metrics.columns
        and pd.notna(r["idx"])
    ):
        idx = int(r["idx"])

        interference = str(
            r["interference"]
        )

        cid = CLASS_ID[interference]

        rows = class_rows[cid]

        if 0 <= idx < len(rows):
            return int(rows[idx])

    raise RuntimeError(
        "Could not map evaluation row "
        "to cached test example."
    )

# ============================================================
# SELECT REPRESENTATIVE CASES
#
# No best-case cherry picking:
# choose FETS result closest to median FETS NMSE
# for each requested condition.
# ============================================================

def representative(
    interference,
    sinr
):
    sub = metrics[
        (metrics["interference"] == interference)
        &
        np.isclose(
            metrics["sinr_db"].astype(float),
            sinr
        )
    ].copy()

    if len(sub) == 0:
        raise RuntimeError(
            f"No rows for "
            f"{interference}, SINR={sinr}"
        )

    median_nmse = (
        sub["nmse_soi_db"].median()
    )

    ix = (
        sub["nmse_soi_db"]
        - median_nmse
    ).abs().idxmin()

    r = metrics.loc[ix]

    return {
        "name": interference,
        "sinr": float(sinr),
        "cache_row":
            cache_row_from_metric(r),
        "eval_idx":
            int(r["idx"]),
        "fets_nmse_from_eval":
            float(r["nmse_soi_db"]),
        "median_fets_nmse":
            float(median_nmse),
    }


CASES = [
    representative(
        "CommSignal3",
        -12.0
    ),
    representative(
        "EMISignal1",
        -12.0
    ),
]

print("\n===== SELECTED REPRESENTATIVE CASES =====")

for case in CASES:
    print(case)

# ============================================================
# DEVICE + MODELS
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

models = {}

for name, ckpt in CKPTS.items():

    model, info = (
        load_coarse_from_checkpoint(
            ckpt,
            device,
            use_ema=True,
        )
    )

    model.eval()
    models[name] = model

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"{name:12s}: "
        f"{params:,} parameters, "
        f"checkpoint step="
        f"{info.get('step')}"
    )

# ============================================================
# HELPERS
# ============================================================

def complex_to_tensor(z):

    arr = np.stack(
        [
            np.real(z),
            np.imag(z),
        ],
        axis=0,
    ).astype(np.float32)

    return (
        torch
        .from_numpy(arr)
        .unsqueeze(0)
        .to(device)
    )


def tensor_to_complex(x):

    x = (
        x.detach()
        .float()
        .cpu()
        .numpy()[0]
    )

    return (
        x[0]
        + 1j * x[1]
    )


def predict(
    model,
    z,
    cid
):

    x = complex_to_tensor(z)

    c = torch.tensor(
        [cid],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():

        try:
            out = model(x, c)

        except TypeError:
            out = model(
                x,
                c,
                None
            )

    return tensor_to_complex(out)


def nmse_db(
    pred,
    gt
):

    numerator = np.sum(
        np.abs(
            pred - gt
        ) ** 2
    )

    denominator = (
        np.sum(
            np.abs(gt) ** 2
        )
        + 1e-12
    )

    return float(
        10.0
        * np.log10(
            numerator
            / denominator
        )
    )


def get_spec(z):

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

    Z = np.fft.fftshift(
        Z,
        axes=0
    )

    return f, t, Z

# ============================================================
# RUN MODELS
# ============================================================

rows = []

print("\n===== SAMPLE-LEVEL RESULTS =====")

for case in CASES:

    i = case["cache_row"]

    mix = np.asarray(
        mixture[i]
    )

    gt = np.asarray(
        target[i]
    )

    cid = int(
        classid[i]
    )

    preds = {}

    for name, model in models.items():

        preds[name] = predict(
            model,
            mix,
            cid
        )

    print(
        f"\n{case['name']}, "
        f"SINR={case['sinr']:g} dB"
    )

    print(
        f"evaluation idx = "
        f"{case['eval_idx']}"
    )

    print(
        f"cache row = "
        f"{case['cache_row']}"
    )

    print(
        f"median FETS NMSE = "
        f"{case['median_fets_nmse']:.3f} dB"
    )

    print(
        f"stored FETS NMSE = "
        f"{case['fets_nmse_from_eval']:.3f} dB"
    )

    for name, pred in preds.items():

        score = nmse_db(
            pred,
            gt
        )

        print(
            f"{name:12s}: "
            f"{score:.3f} dB"
        )

    # Sanity check:
    calculated_fets = nmse_db(
        preds["FETS"],
        gt
    )

    print(
        "FETS stored/recomputed difference = "
        f"{abs(calculated_fets - case['fets_nmse_from_eval']):.6f} dB"
    )

    rows.append({
        **case,
        "mixture": mix,
        "gt": gt,
        "preds": preds,
    })

# ============================================================
# PAPER FIGURE
# ============================================================

COLS = [
    "Observed mixture",
    "Ground truth",
    "Direct FNO",
    "CM-LTS\n(15.00M)",
    "FETS\n(14.84M)",
]

fig, axes = plt.subplots(
    2,
    5,
    figsize=(13.0, 5.1),
    sharex=True,
    sharey=True,
)

plt.subplots_adjust(
    left=0.085,
    right=0.91,
    bottom=0.12,
    top=0.88,
    wspace=0.035,
    hspace=0.11,
)

last_im = None

for r, case in enumerate(rows):

    gt = case["gt"]

    signals = [
        case["mixture"],
        gt,
        case["preds"]["Direct FNO"],
        case["preds"]["CM-LTS"],
        case["preds"]["FETS"],
    ]

    # Ground-truth referenced scale for fair visual comparison.
    _, _, Zgt = get_spec(gt)

    reference = (
        np.max(
            np.abs(Zgt)
        )
        + 1e-12
    )

    for col, z in enumerate(signals):

        ax = axes[r, col]

        f, t, Z = get_spec(z)

        db = 20.0 * np.log10(
            np.abs(Z)
            / reference
            + 1e-10
        )

        last_im = ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=[
                0,
                len(z),
                -0.5,
                0.5
            ],
            vmin=-60,
            vmax=5,
            cmap="viridis",
            interpolation="nearest",
        )

        # Column headings
        if r == 0:
            ax.set_title(
                COLS[col],
                fontsize=10,
                fontweight="bold",
                pad=5,
            )

        # Prediction scores only
        if col >= 2:

            model_name = [
                "Direct FNO",
                "CM-LTS",
                "FETS",
            ][col - 2]

            score = nmse_db(
                case["preds"][model_name],
                gt
            )

            ax.text(
                0.96,
                0.94,
                f"NMSE {score:.2f} dB",
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=7.5,
                color="white",
                bbox=dict(
                    facecolor="black",
                    alpha=0.65,
                    edgecolor="none",
                    pad=2.0,
                ),
            )

        # Row names
        if col == 0:

            ax.set_ylabel(
                f"{case['name']}\n"
                f"SINR = "
                f"{case['sinr']:g} dB",
                fontsize=9,
                fontweight="bold",
            )

        if r == 1:

            ax.set_xlabel(
                "Sample index",
                fontsize=8,
            )

        ax.tick_params(
            labelsize=7,
            length=2,
        )

# Shared frequency label
fig.text(
    0.018,
    0.50,
    "Normalized frequency",
    va="center",
    rotation="vertical",
    fontsize=9,
)

# Shared colorbar
cax = fig.add_axes(
    [
        0.925,
        0.18,
        0.012,
        0.62
    ]
)

cb = fig.colorbar(
    last_im,
    cax=cax,
)

cb.set_label(
    "Magnitude relative to ground-truth peak (dB)",
    fontsize=8.5,
)

cb.ax.tick_params(
    labelsize=7
)

# ============================================================
# SAVE
# ============================================================

png = (
    OUT
    / "qualitative_fets_final.png"
)

pdf = (
    OUT
    / "qualitative_fets_final.pdf"
)

fig.savefig(
    png,
    dpi=400,
    bbox_inches="tight",
)

fig.savefig(
    pdf,
    bbox_inches="tight",
)

plt.close(fig)

print("\n===== SAVED =====")
print(png)
print(pdf)

