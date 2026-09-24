#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
except Exception:
    torch = None


# ============================================================
# Utilities
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_first_existing(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None


def load_metrics_csv(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        print(f"[WARN] metrics csv not found: {path}")
        return None

    df = pd.read_csv(path)
    required = {"interference", "sinr_db", "mse_soi", "nmse_soi_db"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def parse_holdout_from_log(log_path: Path, metric_name: str) -> pd.DataFrame | None:
    """
    Parse lines like:
        HOLDOUT val_mse=0.023352654 at step=90000
        HOLDOUT diffusion_noise_mse=0.30367375 at step=32500
    """
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"[WARN] log not found: {log_path}")
        return None

    pattern = re.compile(
        rf"HOLDOUT\s+{re.escape(metric_name)}=([0-9eE+\-\.]+)\s+at\s+step=(\d+)"
    )

    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                value = float(m.group(1))
                step = int(m.group(2))
                rows.append({"step": step, metric_name: value})

    if not rows:
        print(f"[WARN] no HOLDOUT {metric_name} entries found in {log_path}")
        return None

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


def load_best_val_from_ckpt(ckpt_path: Path):
    if ckpt_path is None or not Path(ckpt_path).exists():
        return None
    if torch is None:
        print("[WARN] torch unavailable, cannot inspect checkpoints.")
        return None

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    extra = ck.get("extra", {})
    best_val = extra.get("best_val_mse", None)
    step = ck.get("step", None)
    return {"best_val_mse": best_val, "step": step}


# ============================================================
# Plotting functions
# ============================================================

def plot_nmse_vs_sinr(df: pd.DataFrame, out_path: Path):
    agg = (
        df.groupby(["interference", "sinr_db"], as_index=False)["nmse_soi_db"]
        .mean()
        .sort_values(["interference", "sinr_db"])
    )

    plt.figure(figsize=(8, 5))
    for name, grp in agg.groupby("interference"):
        grp = grp.sort_values("sinr_db")
        plt.plot(grp["sinr_db"], grp["nmse_soi_db"], marker="o", label=name)

    plt.xlabel("SINR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("Coarse model: NMSE vs SINR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[OK] wrote {out_path}")


def plot_avg_by_class(df: pd.DataFrame, out_path: Path, metric: str = "mse_soi"):
    agg = (
        df.groupby("interference", as_index=False)[metric]
        .mean()
        .sort_values(metric)
    )

    plt.figure(figsize=(7, 5))
    plt.bar(agg["interference"], agg[metric])
    plt.xlabel("Interference class")
    ylabel = "Average MSE" if metric == "mse_soi" else "Average NMSE (dB)"
    plt.ylabel(ylabel)
    title = "Average coarse-model MSE by interference class" if metric == "mse_soi" \
        else "Average coarse-model NMSE by interference class"
    plt.title(title)

    for i, v in enumerate(agg[metric]):
        if metric == "mse_soi":
            txt = f"{v:.4f}"
            offset = max(0.001, 0.03 * agg[metric].max())
        else:
            txt = f"{v:.2f}"
            offset = 0.02 * abs(agg[metric].min())
        plt.text(i, v + (offset if metric == "mse_soi" else 0.0), txt,
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[OK] wrote {out_path}")


def plot_holdout_curve(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str):
    df = df.sort_values(x_col).reset_index(drop=True)
    best_idx = df[y_col].idxmin()
    best_x = float(df.loc[best_idx, x_col])
    best_y = float(df.loc[best_idx, y_col])

    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[y_col], marker="o")
    plt.scatter([best_x], [best_y], s=60)
    plt.annotate(
        f"best = {best_y:.5f} at {int(best_x)}",
        xy=(best_x, best_y),
        xytext=(best_x, best_y + 0.05 * max(df[y_col].max(), 1e-8)),
        arrowprops=dict(arrowstyle="->"),
    )
    plt.xlabel("Step")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[OK] wrote {out_path}")


def plot_heatmap(df: pd.DataFrame, out_path: Path, value_col: str = "nmse_soi_db"):
    agg = (
        df.groupby(["interference", "sinr_db"], as_index=False)[value_col]
        .mean()
    )
    pivot = agg.pivot(index="interference", columns="sinr_db", values=value_col)
    pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))

    plt.figure(figsize=(10, 3.8))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), list(pivot.index))
    plt.xlabel("SINR (dB)")
    plt.ylabel("Interference class")
    title = "Heatmap of average coarse-model NMSE (dB)" if value_col == "nmse_soi_db" \
        else "Heatmap of average coarse-model MSE"
    plt.title(title)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                txt = f"{val:.1f}" if value_col == "nmse_soi_db" else f"{val:.3f}"
                plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.colorbar(im, label="NMSE (dB)" if value_col == "nmse_soi_db" else "MSE")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[OK] wrote {out_path}")


def plot_method_comparison(method_rows: list[dict], out_path: Path):
    """
    Snapshot figure.
    Priority:
      1) final evaluation mean_mse_soi if available
      2) else checkpoint best_val_mse
      3) else pending
    """
    labels = []
    values = []
    sources = []
    pending_names = []

    for row in method_rows:
        labels.append(row["name"])

        if row.get("mean_mse_soi") is not None:
            values.append(row["mean_mse_soi"])
            sources.append("eval")
        elif row.get("best_val_mse") is not None:
            values.append(row["best_val_mse"])
            sources.append("holdout")
        else:
            values.append(np.nan)
            sources.append("pending")
            pending_names.append(row["name"])

    x = np.arange(len(labels))
    values_arr = np.array(values, dtype=float)
    ok = ~np.isnan(values_arr)

    plt.figure(figsize=(11, 5))
    if ok.any():
        plt.bar(x[ok], values_arr[ok])

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Value")
    plt.xlabel("Method")
    plt.title("Method comparison snapshot")

    if ok.any():
        ymax = float(np.nanmax(values_arr[ok]))
    else:
        ymax = 1.0
    plt.ylim(0, ymax * 1.25 + 1e-6)
    plt.grid(True, axis="y", alpha=0.3)

    for i, (v, src) in enumerate(zip(values_arr, sources)):
        if np.isnan(v):
            plt.text(i, ymax * 0.1 + 1e-6, "pending", ha="center", va="bottom", rotation=90)
        else:
            tag = "eval" if src == "eval" else "holdout"
            plt.text(i, v + ymax * 0.03 + 1e-6, f"{v:.4f}\n({tag})",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[OK] wrote {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=str, default=os.environ.get("RUN_ROOT", None),
                    help="Root directory that contains runs/ and evals/")
    ap.add_argument("--log-dir", type=str, default="logs",
                    help="Directory containing slurm/stdout logs")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory for figures")
    args = ap.parse_args()

    if args.run_root is None:
        raise SystemExit("Please provide --run-root or export RUN_ROOT first.")

    run_root = Path(args.run_root).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()

    if args.outdir is None:
        outdir = run_root / "paper_figures"
    else:
        outdir = Path(args.outdir).expanduser().resolve()

    ensure_dir(outdir)

    print(f"[INFO] run_root = {run_root}")
    print(f"[INFO] log_dir  = {log_dir}")
    print(f"[INFO] outdir   = {outdir}")

    # --------------------------------------------------------
    # Key current files
    # --------------------------------------------------------
    coarse_metrics = find_first_existing([
        run_root / "evals/final_coarse_v100/metrics.csv",
        run_root / "evals/proposed_90k_v100/metrics.csv",
    ])

    if coarse_metrics is None:
        print("[WARN] Could not find coarse metrics csv. Figures 1/2/5 may be skipped.")

    coarse_log = find_first_existing([
        log_dir / "rfc-coarse-3178371.out",  # hybrid operator coarse
    ])

    diffusion_log = find_first_existing([
        log_dir / "rfc-diff-3182391.out",    # operator diffusion
    ])

    # Optional / evolving paths — adjust if your naming differs
    method_specs = [
        {
            "name": "Wave-U-Net coarse",
            "eval_csv_candidates": [
                run_root / "evals/baseline_waveunet_v100/metrics.csv",
                run_root / "evals/baseline_waveunet_direct_v100/metrics.csv",
            ],
            "ckpt_candidates": [
                run_root / "baseline_waveunet_v100/coarse/best.pt",
                run_root / "baseline_waveunet_direct_v100/coarse/best.pt",
            ],
        },
        {
            "name": "FNO coarse",
            "eval_csv_candidates": [
                run_root / "evals/baseline_fno_v100/metrics.csv",
            ],
            "ckpt_candidates": [
                run_root / "baseline_fno_v100/coarse/best.pt",
            ],
        },
        {
            "name": "Hybrid operator coarse",
            "eval_csv_candidates": [
                run_root / "evals/final_coarse_v100/metrics.csv",
                run_root / "evals/proposed_90k_v100/metrics.csv",
            ],
            "ckpt_candidates": [
                run_root / "proposed_v100/coarse/best.pt",
            ],
        },
        {
            "name": "Hybrid + residual diffusion",
            "eval_csv_candidates": [
                run_root / "evals/proposed_diff_v100/metrics.csv",
                run_root / "evals/operator_diffusion_v100/metrics.csv",
                run_root / "evals/proposed_v100_diff/metrics.csv",
            ],
            "ckpt_candidates": [
                run_root / "proposed_v100/diffusion/best.pt",
                run_root / "proposed_v100/diff/best.pt",
            ],
        },
        {
            "name": "U-Net + residual diffusion",
            "eval_csv_candidates": [
                run_root / "evals/baseline_unet_diffusion_v100/metrics.csv",
                run_root / "evals/unet_diffusion_v100/metrics.csv",
            ],
            "ckpt_candidates": [
                run_root / "baseline_unet_diffusion_v100/diffusion/best.pt",
                run_root / "baseline_unet_diffusion_v100/diff/best.pt",
            ],
        },
    ]

    # --------------------------------------------------------
    # Figures 1, 2, 5 from coarse metrics
    # --------------------------------------------------------
    if coarse_metrics is not None:
        df_coarse = load_metrics_csv(coarse_metrics)

        if df_coarse is not None:
            plot_nmse_vs_sinr(
                df_coarse,
                outdir / "figure1_nmse_vs_sinr_coarse.png"
            )

            plot_avg_by_class(
                df_coarse,
                outdir / "figure2_average_performance_by_class.png",
                metric="mse_soi",
            )

            plot_heatmap(
                df_coarse,
                outdir / "figure5_heatmap_class_x_sinr.png",
                value_col="nmse_soi_db",
            )

            # Also save summary tables
            per_sinr = (
                df_coarse.groupby(["interference", "sinr_db"], as_index=False)
                .agg(
                    mse_soi=("mse_soi", "mean"),
                    nmse_soi_db=("nmse_soi_db", "mean"),
                    n=("idx", "count") if "idx" in df_coarse.columns else ("sinr_db", "size")
                )
                .sort_values(["interference", "sinr_db"])
            )
            per_sinr.to_csv(outdir / "coarse_per_sinr_summary.csv", index=False)

            avg_by_class = (
                df_coarse.groupby("interference", as_index=False)
                .agg(
                    mean_mse_soi=("mse_soi", "mean"),
                    mean_nmse_soi_db=("nmse_soi_db", "mean"),
                )
                .sort_values("mean_mse_soi")
            )
            avg_by_class.to_csv(outdir / "coarse_avg_by_class.csv", index=False)

            print(f"[OK] wrote {outdir / 'coarse_per_sinr_summary.csv'}")
            print(f"[OK] wrote {outdir / 'coarse_avg_by_class.csv'}")

    # --------------------------------------------------------
    # Figure 3: coarse holdout curve
    # --------------------------------------------------------
    if coarse_log is not None:
        df_holdout = parse_holdout_from_log(coarse_log, "val_mse")
        if df_holdout is not None:
            df_holdout.to_csv(outdir / "coarse_holdout_history.csv", index=False)
            print(f"[OK] wrote {outdir / 'coarse_holdout_history.csv'}")
            plot_holdout_curve(
                df_holdout,
                x_col="step",
                y_col="val_mse",
                out_path=outdir / "figure3_coarse_holdout_curve.png",
                title="Coarse training holdout curve",
            )

    # --------------------------------------------------------
    # Figure 4: diffusion validation curve
    # --------------------------------------------------------
    if diffusion_log is not None:
        df_diff = parse_holdout_from_log(diffusion_log, "diffusion_noise_mse")
        if df_diff is not None:
            df_diff.to_csv(outdir / "diffusion_holdout_history.csv", index=False)
            print(f"[OK] wrote {outdir / 'diffusion_holdout_history.csv'}")
            plot_holdout_curve(
                df_diff,
                x_col="step",
                y_col="diffusion_noise_mse",
                out_path=outdir / "figure4_diffusion_validation_curve.png",
                title="Diffusion validation curve",
            )

    # --------------------------------------------------------
    # Figure 6: method comparison snapshot
    # --------------------------------------------------------
    method_rows = []
    for spec in method_specs:
        row = {"name": spec["name"], "mean_mse_soi": None, "best_val_mse": None}

        eval_csv = find_first_existing(spec["eval_csv_candidates"])
        ckpt = find_first_existing(spec["ckpt_candidates"])

        if eval_csv is not None:
            try:
                df = load_metrics_csv(eval_csv)
                row["mean_mse_soi"] = float(df["mse_soi"].mean())
            except Exception as e:
                print(f"[WARN] failed to read eval csv for {spec['name']}: {e}")

        if ckpt is not None:
            ck_info = load_best_val_from_ckpt(ckpt)
            if ck_info is not None:
                row["best_val_mse"] = ck_info["best_val_mse"]

        method_rows.append(row)

    method_df = pd.DataFrame(method_rows)
    method_df.to_csv(outdir / "method_comparison_snapshot.csv", index=False)
    print(f"[OK] wrote {outdir / 'method_comparison_snapshot.csv'}")

    plot_method_comparison(
        method_rows,
        outdir / "figure6_method_comparison_snapshot.png"
    )

    print("\n[DONE] Figure generation finished.")
    print(f"All outputs are in: {outdir}")


if __name__ == "__main__":
    main()