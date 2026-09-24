#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_run(spec: str):
    if "=" not in spec:
        raise ValueError("Each --run must be LABEL=/path/to/eval_dir")
    label, path = spec.split("=", 1)
    return label, Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="LABEL=/path/to/eval_dir; repeat")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for spec in args.run:
        label, path = parse_run(spec)
        df = pd.read_csv(path / "metrics.csv")
        df["label"] = label
        all_rows.append(df)
    data = pd.concat(all_rows, ignore_index=True)
    data.to_csv(out / "combined_metrics.csv", index=False)

    summary = data.groupby(["label", "interference", "sinr_db"], as_index=False).agg(
        mse=("mse_soi", "mean"), n=("mse_soi", "count")
    )
    summary["mse_db"] = 10 * np.log10(summary["mse"].clip(lower=1e-12))
    summary.to_csv(out / "combined_summary.csv", index=False)

    for intr in sorted(summary.interference.unique()):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        sub = summary[summary.interference == intr]
        for label in sub.label.unique():
            d = sub[sub.label == label].sort_values("sinr_db")
            ax.plot(d.sinr_db, d.mse_db, marker="o", label=label)
        ax.set_xlabel("SINR (dB)")
        ax.set_ylabel("SOI MSE (dB)")
        ax.set_title(intr)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"mse_vs_sinr_{intr}.pdf")
        fig.savefig(out / f"mse_vs_sinr_{intr}.png", dpi=220)
        plt.close(fig)

    overall = data.groupby("label", as_index=False).agg(mean_mse=("mse_soi", "mean"))
    overall["mean_mse_db"] = 10 * np.log10(overall.mean_mse.clip(lower=1e-12))
    overall.to_csv(out / "overall.csv", index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
