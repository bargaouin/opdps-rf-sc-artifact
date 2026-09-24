#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import stft
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ============================================================
# RF helpers
# ============================================================

CLASS_TO_ID = {
    "EMISignal1": 0,
    "CommSignal3": 1,
}


def as_complex(x: np.ndarray) -> np.ndarray:
    """
    Accept common complex-I/Q representations:

      [L] complex
      [2, L] real I/Q
      [L, 2] real I/Q

    and return [L] complex64.
    """
    x = np.asarray(x)

    if np.iscomplexobj(x):
        return x.astype(np.complex64, copy=False).squeeze()

    x = np.squeeze(x)

    if x.ndim == 2:
        if x.shape[0] == 2:
            return (
                x[0].astype(np.float32)
                + 1j * x[1].astype(np.float32)
            )

        if x.shape[-1] == 2:
            return (
                x[..., 0].astype(np.float32)
                + 1j * x[..., 1].astype(np.float32)
            )

    raise ValueError(
        f"Cannot convert array of shape {x.shape} and dtype {x.dtype} "
        "to a complex RF waveform."
    )


def load_prediction_array(path: Path):
    """
    Supports .npy and .npz predictions.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")

    if path.suffix == ".npz":
        z = np.load(path)

        candidates = [
            "predictions",
            "prediction",
            "pred",
            "estimates",
            "estimate",
            "outputs",
            "output",
            "x_hat",
            "shat",
            "s_hat",
        ]

        for k in candidates:
            if k in z:
                return z[k]

        if len(z.files) == 1:
            return z[z.files[0]]

        raise ValueError(
            f"{path} contains keys {z.files}. "
            "I do not know which one is the prediction."
        )

    raise ValueError(f"Unsupported prediction format: {path}")


# ============================================================
# Example selection
# ============================================================

def choose_representative(
    metrics: pd.DataFrame,
    interference: str,
    sinr_db: float,
) -> pd.Series:
    """
    Select a representative sample at a fixed interference/SINR.

    Rather than choosing the visually best example, use the sample
    whose coarse-model MSE is closest to the median MSE of that cell.
    """

    d = metrics[
        (metrics["interference"] == interference)
        & np.isclose(metrics["sinr_db"], sinr_db)
    ].copy()

    if len(d) == 0:
        available = (
            metrics[metrics["interference"] == interference]["sinr_db"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        raise RuntimeError(
            f"No examples for {interference}, SINR={sinr_db}. "
            f"Available SINRs: {available}"
        )

    median_mse = d["mse_soi"].median()

    d["distance_from_median"] = np.abs(
        d["mse_soi"] - median_mse
    )

    return d.sort_values("distance_from_median").iloc[0]


def find_cache_row(
    cache_class_ids: np.ndarray,
    cache_indices: np.ndarray,
    interference: str,
    local_idx: int,
) -> int:

    class_id = CLASS_TO_ID[interference]

    where = np.where(
        (cache_class_ids == class_id)
        & (cache_indices == local_idx)
    )[0]

    if len(where) != 1:
        raise RuntimeError(
            f"Expected one cache match for "
            f"{interference}, idx={local_idx}; got {len(where)}"
        )

    return int(where[0])


# ============================================================
# Spectrogram
# ============================================================

def complex_spectrogram(
    x: np.ndarray,
    sample_rate: float | None,
    nperseg: int,
    noverlap: int,
):

    x = as_complex(x)

    fs = sample_rate if sample_rate is not None else 1.0

    f, t, Z = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        return_onesided=False,
        padded=False,
        boundary=None,
    )

    # Center negative/positive frequency.
    f = np.fft.fftshift(f)
    Z = np.fft.fftshift(Z, axes=0)

    mag = np.abs(Z)

    return f, t, mag


def mag_to_db(mag, reference):
    eps = 1e-12
    return 20.0 * np.log10(
        np.maximum(mag, eps) / max(reference, eps)
    )


# ============================================================
# Main qualitative plot
# ============================================================

def make_figure(
    cache_dir: Path,
    metrics_path: Path,
    methods: list[tuple[str, Path]],
    cases: list[tuple[str, float]],
    output: Path,
    sample_rate: float | None,
    nperseg: int,
    noverlap: int,
    db_floor: float,
    db_ceil: float,
    inset_samples: int,
):

    # --------------------------------------------------------
    # Load RFChallenge cached validation data
    # --------------------------------------------------------

    mixture_path = cache_dir / "sep_val_mixture.npy"
    target_path = cache_dir / "sep_val_target.npy"
    class_path = cache_dir / "sep_val_class_id.npy"
    idx_path = cache_dir / "sep_val_idx.npy"

    for p in [
        mixture_path,
        target_path,
        class_path,
        idx_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    mixtures = np.load(mixture_path, mmap_mode="r")
    targets = np.load(target_path, mmap_mode="r")
    class_ids = np.load(class_path, mmap_mode="r")
    cache_indices = np.load(idx_path, mmap_mode="r")

    metrics = pd.read_csv(metrics_path)

    print("Cache:")
    print(" mixture:", mixtures.shape, mixtures.dtype)
    print(" target :", targets.shape, targets.dtype)
    print(" classes:", class_ids.shape)
    print(" indices:", cache_indices.shape)

    # --------------------------------------------------------
    # Load model predictions
    # --------------------------------------------------------

    loaded_methods = []

    for label, path in methods:
        try:
            arr = load_prediction_array(path)

            if len(arr) != len(mixtures):
                print(
                    f"[WARN] skipping {label}: "
                    f"prediction N={len(arr)} while cache N={len(mixtures)}"
                )
                continue

            loaded_methods.append((label, arr))
            print(
                f"[OK] {label}: {path} "
                f"shape={arr.shape} dtype={arr.dtype}"
            )

        except Exception as exc:
            print(f"[WARN] skipping {label}: {exc}")

    columns = [
        ("Observed mixture", None),
        ("Ground-truth SOI", None),
    ]

    columns.extend(
        [(label, arr) for label, arr in loaded_methods]
    )

    # --------------------------------------------------------
    # Select examples
    # --------------------------------------------------------

    selected = []

    for interference, sinr in cases:

        result = choose_representative(
            metrics,
            interference,
            sinr,
        )

        local_idx = int(result["idx"])

        cache_row = find_cache_row(
            class_ids,
            cache_indices,
            interference,
            local_idx,
        )

        selected.append(
            {
                "interference": interference,
                "sinr": float(sinr),
                "local_idx": local_idx,
                "cache_row": cache_row,
                "mse": float(result["mse_soi"]),
                "nmse": float(result["nmse_soi_db"]),
            }
        )

    print("\nSelected examples:")

    for s in selected:
        print(
            f"  {s['interference']:12s} "
            f"SINR={s['sinr']:5.1f} dB "
            f"idx={s['local_idx']:4d} "
            f"cache_row={s['cache_row']:4d} "
            f"coarse NMSE={s['nmse']:.2f} dB"
        )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    nrows = len(selected)
    ncols = len(columns)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.05 * ncols, 2.8 * nrows),
        squeeze=False,
    )

    for row_i, example in enumerate(selected):

        row = example["cache_row"]

        mixture = as_complex(mixtures[row])
        target = as_complex(targets[row])

        signals = [
            mixture,
            target,
        ]

        for _, arr in loaded_methods:
            signals.append(as_complex(arr[row]))

        # ----------------------------------------------------
        # One common spectral reference for the whole row.
        #
        # This is deliberate:
        # DO NOT independently normalize each method.
        # Otherwise amplitude errors become invisible.
        # ----------------------------------------------------

        _, _, gt_mag = complex_spectrogram(
            target,
            sample_rate,
            nperseg,
            noverlap,
        )

        reference = float(np.max(gt_mag))

        # Same waveform y limits throughout a row
        amplitude_limit = max(
            np.max(np.abs(np.real(mixture))),
            np.max(np.abs(np.imag(mixture))),
            np.max(np.abs(np.real(target))),
            np.max(np.abs(np.imag(target))),
        )

        amplitude_limit *= 1.05

        for col_i, (column, signal) in enumerate(
            zip(columns, signals)
        ):

            title, _ = column
            ax = axes[row_i, col_i]

            f, t, mag = complex_spectrogram(
                signal,
                sample_rate,
                nperseg,
                noverlap,
            )

            spec_db = mag_to_db(mag, reference)

            if sample_rate is not None:
                t_plot = t * 1e3
                f_plot = f / 1e6
            else:
                t_plot = t
                f_plot = f

            mesh = ax.pcolormesh(
                t_plot,
                f_plot,
                spec_db,
                shading="auto",
                vmin=db_floor,
                vmax=db_ceil,
            )

            # Column titles only at top
            if row_i == 0:
                ax.set_title(
                    title,
                    fontsize=11,
                    fontweight="bold",
                )

            # ------------------------------------------------
            # Row label
            # ------------------------------------------------

            if col_i == 0:
                row_label = (
                    f"{example['interference']}\n"
                    f"SINR = {example['sinr']:g} dB"
                )

                ax.set_ylabel(
                    row_label,
                    fontsize=10,
                    fontweight="bold",
                )
            else:
                ax.set_yticklabels([])

            if row_i == nrows - 1:
                if sample_rate is not None:
                    ax.set_xlabel("Time (ms)")
                else:
                    ax.set_xlabel("Normalized time")
            else:
                ax.set_xticklabels([])

            # ------------------------------------------------
            # I/Q waveform inset
            # ------------------------------------------------

            inset = inset_axes(
                ax,
                width="38%",
                height="28%",
                loc="upper right",
                borderpad=0.65,
            )

            n = min(inset_samples, len(signal))

            inset.plot(np.real(signal[:n]), linewidth=0.55)
            inset.plot(np.imag(signal[:n]), linewidth=0.55)

            inset.set_xlim(0, n - 1)
            inset.set_ylim(
                -amplitude_limit,
                amplitude_limit,
            )

            inset.set_xticks([])
            inset.set_yticks([])

            # ------------------------------------------------
            # Optional NMSE for method predictions
            # ------------------------------------------------

            if col_i >= 2:
                err = signal - target

                nmse = 10.0 * np.log10(
                    np.sum(np.abs(err) ** 2)
                    / np.maximum(
                        np.sum(np.abs(target) ** 2),
                        1e-12,
                    )
                )

                ax.text(
                    0.02,
                    0.03,
                    f"NMSE {nmse:.1f} dB",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.85,
                        linewidth=0.5,
                    ),
                )

    # --------------------------------------------------------
    # Shared colorbar
    # --------------------------------------------------------

    cbar = fig.colorbar(
        mesh,
        ax=axes,
        fraction=0.018,
        pad=0.01,
    )

    cbar.set_label(
        "Magnitude relative to ground-truth peak (dB)"
    )

    fig.suptitle(
        "Qualitative RF Separation Examples",
        fontsize=14,
        fontweight="bold",
    )

    # subplots_adjust works more reliably than tight_layout
    # with inset axes + shared colorbar.
    fig.subplots_adjust(
        left=0.08,
        right=0.91,
        bottom=0.08,
        top=0.91,
        wspace=0.08,
        hspace=0.10,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    pdf_output = output.with_suffix(".pdf")

    fig.savefig(
        pdf_output,
        bbox_inches="tight",
    )

    print(f"\n[OK] PNG: {output}")
    print(f"[OK] PDF: {pdf_output}")

    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def parse_method(value: str):
    """
    Syntax:
        --method "Hybrid coarse=/path/predictions.npy"
    """

    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Method syntax must be LABEL=/path/to/predictions.npy"
        )

    label, path = value.split("=", 1)

    return label.strip(), Path(path).expanduser()


def parse_case(value: str):
    """
    Syntax:
        --case "EMISignal1:3"
        --case "CommSignal3:-12"
    """

    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Case syntax must be INTERFERENCE:SINR"
        )

    name, sinr = value.rsplit(":", 1)

    if name not in CLASS_TO_ID:
        raise argparse.ArgumentTypeError(
            f"Unknown interference {name}. "
            f"Choose from {list(CLASS_TO_ID)}"
        )

    return name, float(sinr)


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--metrics",
        required=True,
        type=Path,
        help="Reference evaluation metrics.csv used to select "
             "representative examples.",
    )

    p.add_argument(
        "--method",
        action="append",
        type=parse_method,
        default=[],
        help=(
            'Repeatable. Example: '
            '--method "Hybrid coarse=/path/predictions.npy"'
        ),
    )

    p.add_argument(
        "--case",
        action="append",
        type=parse_case,
        default=None,
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("qualitative_rf_comparison.png"),
    )

    p.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help=(
            "Sample rate in Hz. "
            "If omitted, normalized frequency/time are used."
        ),
    )

    p.add_argument(
        "--nperseg",
        type=int,
        default=512,
    )

    p.add_argument(
        "--noverlap",
        type=int,
        default=384,
    )

    p.add_argument(
        "--db-floor",
        type=float,
        default=-60.0,
    )

    p.add_argument(
        "--db-ceil",
        type=float,
        default=6.0,
    )

    p.add_argument(
        "--inset-samples",
        type=int,
        default=1500,
    )

    args = p.parse_args()

    if args.case is None:
        cases = [
            ("EMISignal1", 3.0),
            ("CommSignal3", -4.5),
            ("CommSignal3", -12.0),
        ]
    else:
        cases = args.case

    make_figure(
        cache_dir=args.cache_dir,
        metrics_path=args.metrics,
        methods=args.method,
        cases=cases,
        output=args.output,
        sample_rate=args.sample_rate,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        db_floor=args.db_floor,
        db_ceil=args.db_ceil,
        inset_samples=args.inset_samples,
    )


if __name__ == "__main__":
    main()