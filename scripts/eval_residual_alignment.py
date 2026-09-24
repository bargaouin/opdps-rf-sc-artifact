from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from opdps_rf_sc.diffusion import DiffusionSchedule, sample_residual_ensemble
from opdps_rf_sc.model_io import (
    load_coarse_from_checkpoint,
    load_diffusion_from_checkpoint,
)


def complex_to_ri(x: np.ndarray) -> torch.Tensor:
    """Complex [B,L] -> real I/Q [B,2,L]."""
    return torch.from_numpy(
        np.stack([x.real, x.imag], axis=1).astype(np.float32)
    )


def mse_per_example(x: torch.Tensor) -> torch.Tensor:
    return x.square().mean(dim=(1, 2))


def real_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # In I/Q representation this is the real complex inner product.
    return (a * b).sum(dim=(1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--coarse-checkpoint", required=True)
    ap.add_argument("--diffusion-checkpoint", required=True)
    ap.add_argument("--residual-stats", required=True)
    ap.add_argument("--coarse-metrics", required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--sinr", type=float, default=3.0)
    ap.add_argument("--per-class", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--solver-steps", type=int, default=20)
    ap.add_argument("--sampler", default="dpm")
    ap.add_argument("--k-values", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0],
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cache = Path(args.cache_dir)

    mix_np = np.load(cache / "sep_val_mixture.npy", mmap_mode="r")
    tgt_np = np.load(cache / "sep_val_target.npy", mmap_mode="r")
    cls_np = np.load(cache / "sep_val_class_id.npy", mmap_mode="r")

    metrics = pd.read_csv(args.coarse_metrics)

    print("cache shapes:")
    print(" mixture:", mix_np.shape, mix_np.dtype)
    print(" target :", tgt_np.shape, tgt_np.dtype)
    print(" class  :", cls_np.shape, cls_np.dtype)
    print("metrics:", metrics.shape)

    if len(metrics) != len(mix_np):
        raise RuntimeError(
            f"metrics rows ({len(metrics)}) != cache examples ({len(mix_np)}). "
            "This script assumes eval metrics preserve cache order."
        )

    # Select exactly the same style of subset used in the previous 64-example test:
    # 32 examples/class at SINR = 3 dB.
    selected = []
    mask_sinr = np.isclose(metrics["sinr_db"].to_numpy(), args.sinr)

    for name in metrics.loc[mask_sinr, "interference"].drop_duplicates():
        rows = metrics.index[
            mask_sinr & (metrics["interference"].to_numpy() == name)
        ].to_numpy()[: args.per_class]

        if len(rows) < args.per_class:
            raise RuntimeError(
                f"Only found {len(rows)} examples for {name} at SINR={args.sinr}"
            )

        selected.extend(rows.tolist())

    selected = np.asarray(selected, dtype=np.int64)

    print("\nSelected subset:")
    print(metrics.iloc[selected].groupby(["interference", "sinr_db"]).size())

    coarse, _ = load_coarse_from_checkpoint(
        args.coarse_checkpoint, device, use_ema=True
    )
    diffusion, _ = load_diffusion_from_checkpoint(
        args.diffusion_checkpoint, device, use_ema=True
    )

    coarse.eval()
    diffusion.eval()

    stats = json.loads(Path(args.residual_stats).read_text())
    scales = torch.tensor(
        stats["scales"], dtype=torch.float32, device=device
    )

    timesteps = int(cfg.get("diffusion", {}).get("timesteps", 1000))
    schedule = DiffusionSchedule(timesteps).to(device)

    all_alignment_rows = []
    all_sweep_rows = []
    summaries = {}

    for K in args.k_values:
        print("\n" + "=" * 80)
        print(f"POSTERIOR SAMPLES K={K}")
        print("=" * 80)

        k_rows = []

        for start in range(0, len(selected), args.batch_size):
            rows = selected[start:start + args.batch_size]

            mix = complex_to_ri(np.asarray(mix_np[rows])).to(device)
            target = complex_to_ri(np.asarray(tgt_np[rows])).to(device)
            class_id = torch.from_numpy(
                np.asarray(cls_np[rows]).astype(np.int64)
            ).to(device)

            with torch.no_grad():
                coarse_pred = coarse(mix, class_id).float()

                # blend=1 means:
                # refined = coarse + physical-scale * posterior mean residual
                refined = sample_residual_ensemble(
                    diffusion,
                    mix,
                    coarse_pred,
                    class_id,
                    schedule,
                    scales,
                    num_samples=K,
                    steps=args.solver_steps,
                    sampler=args.sampler,
                    blend=1.0,
                    seed=args.seed,
                ).float()

            # True residual still missing after Stage 1.
            d = target - coarse_pred

            # Learned Stage-2 correction in physical I/Q units.
            c = refined - coarse_pred

            dot = real_inner(d, c)
            d2 = real_inner(d, d)
            c2 = real_inner(c, c)

            eps = 1e-12

            cosine = dot / torch.sqrt(
                torch.clamp(d2 * c2, min=eps)
            )

            beta_oracle = dot / torch.clamp(c2, min=eps)

            error_norm = torch.sqrt(torch.clamp(d2, min=eps))
            corr_norm = torch.sqrt(torch.clamp(c2, min=eps))
            norm_ratio = corr_norm / error_norm

            coarse_mse = mse_per_example(d)

            for j, row_idx in enumerate(rows):
                meta = metrics.iloc[int(row_idx)]

                k_rows.append(
                    {
                        "cache_row": int(row_idx),
                        "interference": str(meta["interference"]),
                        "sinr_db": float(meta["sinr_db"]),
                        "idx": int(meta["idx"]),
                        "K": int(K),
                        "coarse_mse": float(coarse_mse[j].cpu()),
                        "dot_dc": float(dot[j].cpu()),
                        "true_residual_energy": float(d2[j].cpu()),
                        "correction_energy": float(c2[j].cpu()),
                        "cosine_alignment": float(cosine[j].cpu()),
                        "oracle_beta": float(beta_oracle[j].cpu()),
                        "correction_to_error_norm": float(norm_ratio[j].cpu()),
                    }
                )

            # Evaluate arbitrary beta values WITHOUT another diffusion pass.
            for beta in args.betas:
                pred = coarse_pred + float(beta) * c
                err = pred - target
                mse = mse_per_example(err)

                for j, row_idx in enumerate(rows):
                    meta = metrics.iloc[int(row_idx)]

                    # NMSE denominator = target signal power.
                    target_pow = (
                        target[j].square().mean().clamp_min(1e-12)
                    )
                    nmse_db = 10.0 * torch.log10(
                        mse[j].clamp_min(1e-12) / target_pow
                    )

                    all_sweep_rows.append(
                        {
                            "cache_row": int(row_idx),
                            "interference": str(meta["interference"]),
                            "sinr_db": float(meta["sinr_db"]),
                            "idx": int(meta["idx"]),
                            "K": int(K),
                            "beta": float(beta),
                            "mse": float(mse[j].cpu()),
                            "nmse_db": float(nmse_db.cpu()),
                        }
                    )

            print(
                f"processed {min(start + len(rows), len(selected))}/{len(selected)}"
            )

        kdf = pd.DataFrame(k_rows)
        all_alignment_rows.extend(k_rows)

        # Analytic global optimum beta over the selected subset.
        dot_total = kdf["dot_dc"].sum()
        c2_total = kdf["correction_energy"].sum()
        beta_global = dot_total / max(c2_total, 1e-12)

        summary = {
            "K": int(K),
            "examples": int(len(kdf)),
            "mean_cosine_alignment": float(kdf["cosine_alignment"].mean()),
            "median_cosine_alignment": float(kdf["cosine_alignment"].median()),
            "fraction_positive_alignment": float(
                (kdf["cosine_alignment"] > 0).mean()
            ),
            "fraction_negative_alignment": float(
                (kdf["cosine_alignment"] < 0).mean()
            ),
            "mean_oracle_beta": float(kdf["oracle_beta"].mean()),
            "median_oracle_beta": float(kdf["oracle_beta"].median()),
            "global_oracle_beta": float(beta_global),
            "mean_correction_to_error_norm": float(
                kdf["correction_to_error_norm"].mean()
            ),
            "median_correction_to_error_norm": float(
                kdf["correction_to_error_norm"].median()
            ),
        }

        summaries[str(K)] = summary

        print(json.dumps(summary, indent=2))

        print("\nBY CLASS:")
        print(
            kdf.groupby("interference")[
                [
                    "cosine_alignment",
                    "oracle_beta",
                    "correction_to_error_norm",
                ]
            ]
            .mean()
            .to_string()
        )

    adf = pd.DataFrame(all_alignment_rows)
    sdf = pd.DataFrame(all_sweep_rows)

    adf.to_csv(outdir / "residual_alignment.csv", index=False)
    sdf.to_csv(outdir / "beta_sweep.csv", index=False)

    sweep_summary = (
        sdf.groupby(["K", "beta"], as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            mean_nmse_db=("nmse_db", "mean"),
        )
    )
    sweep_summary.to_csv(outdir / "beta_sweep_summary.csv", index=False)

    class_summary = (
        adf.groupby(["K", "interference"], as_index=False)
        .agg(
            mean_cosine=("cosine_alignment", "mean"),
            median_cosine=("cosine_alignment", "median"),
            mean_oracle_beta=("oracle_beta", "mean"),
            median_oracle_beta=("oracle_beta", "median"),
            mean_norm_ratio=("correction_to_error_norm", "mean"),
        )
    )
    class_summary.to_csv(outdir / "alignment_by_class.csv", index=False)

    (outdir / "summary.json").write_text(
        json.dumps(summaries, indent=2)
    )

    print("\n" + "=" * 80)
    print("BETA SWEEP SUMMARY")
    print("=" * 80)
    print(sweep_summary.to_string(index=False))

    print("\nSaved:")
    for p in sorted(outdir.iterdir()):
        print(" ", p)


if __name__ == "__main__":
    main()
