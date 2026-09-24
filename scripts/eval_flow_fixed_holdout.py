#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from opdps_rf_sc.config import load_yaml
from opdps_rf_sc.data import MixtureBatchGenerator
from opdps_rf_sc.model_io import load_coarse_from_checkpoint
from opdps_rf_sc.models import build_separator
from opdps_rf_sc.rectified_flow import sample_rectified_flow
from opdps_rf_sc.train_utils import configure_torch, resolve_device


def load_scales(path, device):
    obj = json.loads(Path(path).read_text())
    return torch.tensor(
        obj["scales"],
        dtype=torch.float32,
        device=device,
    )


def load_flow(path, device):
    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = ck["config"]

    flow = build_separator(
        cfg["flow"]["model"],
        in_channels=6,
        out_channels=2,
        time_conditioned=True,
    )

    flow.load_state_dict(
        ck.get("flow_ema") or ck["flow_model"]
    )
    flow.to(device).eval()

    source = None

    if ck["mode"] == "mixed":
        source = build_separator(
            cfg["source"]["model"],
            in_channels=4,
            out_channels=4,
            time_conditioned=False,
        )

        source.load_state_dict(
            ck.get("source_ema") or ck["source_model"]
        )

        source.to(device).eval()

    return flow, source, ck


def metrics(pred, target):
    err = (
        (pred - target)
        .square()
        .sum(dim=1)
        .mean(dim=1)
    )

    power = (
        target.square()
        .sum(dim=1)
        .mean(dim=1)
        .clamp_min(1e-12)
    )

    nmse = 10.0 * torch.log10(
        err.clamp_min(1e-12) / power
    )

    return err, nmse


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--coarse-checkpoint", required=True)
    ap.add_argument("--residual-stats", required=True)

    ap.add_argument("--gaussian", required=True)
    ap.add_argument("--anchor", required=True)

    ap.add_argument("--examples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=616161)

    ap.add_argument("--solver", default="heun")
    ap.add_argument("--ode-steps", type=int, default=4)
    ap.add_argument("--source-weight", type=float, default=1.0)

    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    configure_torch()
    device = resolve_device(args.device)

    cfg = load_yaml(args.config)

    coarse, _ = load_coarse_from_checkpoint(
        args.coarse_checkpoint,
        device,
        use_ema=True,
    )

    scales = load_scales(
        args.residual_stats,
        device,
    )

    gflow, _, gck = load_flow(
        args.gaussian,
        device,
    )

    aflow, asource, ack = load_flow(
        args.anchor,
        device,
    )

    print(
        "Gaussian checkpoint step:",
        gck["step"],
    )
    print(
        "Anchor checkpoint step:",
        ack["step"],
    )

    tcfg = cfg["train"]

    gen = MixtureBatchGenerator(
        cache_dir=args.cache_dir,
        length=int(tcfg.get("length", 40960)),
        continuous_sinr_prob=float(
            tcfg.get("continuous_sinr_prob", 0.2)
        ),
        continuous_sinr_range=(
            float(tcfg.get("continuous_sinr_min", -15)),
            float(tcfg.get("continuous_sinr_max", 5)),
        ),
        low_sinr_bias=float(
            tcfg.get("low_sinr_bias", 0.35)
        ),
        common_phase_aug_prob=0.0,
        split="holdout",
        seed=args.seed,
    )

    results = {
        "coarse_mse": [],
        "coarse_nmse": [],
        "gaussian_mse": [],
        "gaussian_nmse": [],
        "anchor_mse": [],
        "anchor_nmse": [],
    }

    done = 0

    while done < args.examples:
        bs = min(
            args.batch_size,
            args.examples - done,
        )

        batch = gen.batch(bs, device)

        with torch.no_grad():
            coarse_pred = coarse(
                batch.mixture,
                batch.class_id,
            ).float()

            c_mse, c_nmse = metrics(
                coarse_pred,
                batch.target.float(),
            )

            # Same initial random seeds for reproducibility.
            gg = torch.Generator(device=device)
            gg.manual_seed(args.seed + done)

            r_g = sample_rectified_flow(
                gflow,
                batch.mixture,
                coarse_pred,
                batch.class_id,
                mode="gaussian",
                solver=args.solver,
                steps=args.ode_steps,
                generator=gg,
            )

            scale = scales[
                batch.class_id
            ].reshape(-1, 1, 1)

            pred_g = coarse_pred + scale * r_g

            g_mse, g_nmse = metrics(
                pred_g,
                batch.target.float(),
            )

            ag = torch.Generator(device=device)
            ag.manual_seed(args.seed + done)

            r_a = sample_rectified_flow(
                aflow,
                batch.mixture,
                coarse_pred,
                batch.class_id,
                mode="mixed",
                source_model=asource,
                source_weight=args.source_weight,
                solver=args.solver,
                steps=args.ode_steps,
                generator=ag,
                logvar_min=float(
                    ack["config"]["source"].get(
                        "logvar_min", -6.0
                    )
                ),
                logvar_max=float(
                    ack["config"]["source"].get(
                        "logvar_max", 3.0
                    )
                ),
                time_scale=float(
                    ack["config"]["flow"].get(
                        "time_scale", 1000.0
                    )
                ),
            )

            pred_a = coarse_pred + scale * r_a

            a_mse, a_nmse = metrics(
                pred_a,
                batch.target.float(),
            )

        for key, val in [
            ("coarse_mse", c_mse),
            ("coarse_nmse", c_nmse),
            ("gaussian_mse", g_mse),
            ("gaussian_nmse", g_nmse),
            ("anchor_mse", a_mse),
            ("anchor_nmse", a_nmse),
        ]:
            results[key].extend(
                val.detach().cpu().tolist()
            )

        done += bs

        if done % 32 == 0 or done == args.examples:
            print(
                f"processed {done}/{args.examples}",
                flush=True,
            )

    def summarize(prefix):
        mse = np.asarray(
            results[f"{prefix}_mse"]
        )

        nmse = np.asarray(
            results[f"{prefix}_nmse"]
        )

        return {
            "mean_mse": float(mse.mean()),
            "median_mse": float(np.median(mse)),
            "mean_nmse_db": float(nmse.mean()),
            "median_nmse_db": float(np.median(nmse)),
        }

    summary = {
        "examples": args.examples,
        "solver": args.solver,
        "ode_steps": args.ode_steps,
        "source_weight": args.source_weight,
        "coarse": summarize("coarse"),
        "gaussian_rf": summarize("gaussian"),
        "anchorflow": summarize("anchor"),
    }

    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
