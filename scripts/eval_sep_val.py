#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from opdps_rf_sc.cache import SepValCache
from opdps_rf_sc.complex_utils import np_complex_to_2ch, twoch_to_np_complex
from opdps_rf_sc.config import load_yaml
from opdps_rf_sc.constants import (
    ID_TO_INTERFERENCE,
    INTERFERENCE_TO_ID,
    OFFICIAL_SINR_LEVELS_DESC,
    SEPARATION_INTERFERERS,
)
from opdps_rf_sc.diffusion import DiffusionSchedule, sample_residual_ensemble
from opdps_rf_sc.metrics import mse_complex_np, nmse_db_complex_np
from opdps_rf_sc.model_io import load_coarse_from_checkpoint, load_diffusion_from_checkpoint
from opdps_rf_sc.rfchallenge import RFCAdapter
from opdps_rf_sc.train_utils import configure_torch, resolve_device


def infer_sinr_label(idx: int) -> float:
    block = idx // 100
    if block >= len(OFFICIAL_SINR_LEVELS_DESC):
        return float("nan")
    return float(OFFICIAL_SINR_LEVELS_DESC[block])


def load_blend(blend_arg: float | None, blend_json: str | None, cfg: dict) -> float:
    if blend_arg is not None:
        return float(blend_arg)
    if blend_json:
        return float(json.loads(Path(blend_json).read_text())["best_blend"])
    return float(cfg.get("inference", {}).get("blend", 1.0))


def iter_cache_examples(cache_dir: str, max_examples: int):
    cache = SepValCache(cache_dir)
    for cid, intr in ID_TO_INTERFERENCE.items():
        rows = np.flatnonzero(np.asarray(cache.class_id) == cid)
        rows = rows[np.argsort(np.asarray(cache.idx)[rows])]
        rows = rows[: min(max_examples, rows.size)]
        for row in rows:
            idx = int(cache.idx[row])
            y = np.asarray(cache.mixture[row], dtype=np.complex64)
            s = np.asarray(cache.target[row], dtype=np.complex64)
            yield intr, idx, y, s


def iter_rfcutils_examples(rf_root: str, max_examples: int):
    adapter = RFCAdapter(rf_root)
    for intr in SEPARATION_INTERFERERS:
        for idx in range(min(1100, max_examples)):
            y, _ = adapter.load_sample(idx, "sep_val", intr)
            s, _, b, _ = adapter.load_components(idx, "sep_val", intr)
            y = np.asarray(y, dtype=np.complex64).reshape(-1)
            s = np.asarray(s, dtype=np.complex64).reshape(-1)
            b = np.asarray(b, dtype=np.complex64).reshape(-1)
            if not np.allclose(y, s + b, rtol=2e-5, atol=2e-5):
                raise ValueError(f"Official sample {intr}/{idx} fails y=s+b consistency check")
            yield intr, idx, y, s


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache-dir", help="Prepared cache containing sep_val arrays (recommended)")
    src.add_argument("--rf-root", help="Fallback: read sep_val through the official rfcutils directly")
    ap.add_argument("--config", required=True)
    ap.add_argument("--coarse-checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--diffusion-checkpoint", default=None)
    ap.add_argument("--residual-stats", default=None)
    ap.add_argument("--blend-json", default=None)
    ap.add_argument("--blend", type=float, default=None)
    ap.add_argument("--sampler", choices=["dpm", "ddim"], default=None)
    ap.add_argument("--solver-steps", type=int, default=None)
    ap.add_argument("--posterior-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=1100, help="Per interference type; use smaller for a quick check")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    configure_torch()
    device = resolve_device(args.device)
    coarse, _ = load_coarse_from_checkpoint(args.coarse_checkpoint, device, use_ema=True)

    diffusion = schedule = scales = None
    blend = 0.0
    sampler = None
    solver_steps = None
    posterior_samples = None
    if args.diffusion_checkpoint:
        if not args.residual_stats:
            raise ValueError("--residual-stats is required with --diffusion-checkpoint")
        diffusion, _ = load_diffusion_from_checkpoint(args.diffusion_checkpoint, device, use_ema=True)
        stats = json.loads(Path(args.residual_stats).read_text())
        scales = torch.tensor(stats["scales"], device=device, dtype=torch.float32)
        schedule = DiffusionSchedule(int(cfg["diffusion"].get("timesteps", 1000))).to(device)
        icfg = cfg.get("inference", {})
        blend = load_blend(args.blend, args.blend_json, cfg)
        sampler = args.sampler or icfg.get("sampler", "dpm")
        solver_steps = int(args.solver_steps or icfg.get("solver_steps", 20))
        posterior_samples = int(args.posterior_samples or icfg.get("posterior_samples", 4))

    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "metrics.csv"
    done = set()
    rows = []
    if args.resume and csv_path.exists():
        old = pd.read_csv(csv_path)
        rows = old.to_dict("records")
        done = {(str(r["interference"]), int(r["idx"])) for r in rows}
        print(f"Resuming with {len(done)} examples already present")

    source_iter = (
        iter_cache_examples(args.cache_dir, args.max_examples)
        if args.cache_dir
        else iter_rfcutils_examples(args.rf_root, args.max_examples)
    )
    examples = [e for e in source_iter if (e[0], e[1]) not in done]
    method = "operator_diffusion" if diffusion is not None else "coarse"

    pos = 0
    while pos < len(examples):
        intr0 = examples[pos][0]
        end = pos
        while end < len(examples) and examples[end][0] == intr0 and end - pos < args.batch_size:
            end += 1
        chunk = examples[pos:end]
        intrs, ids, mix_np, target_np = zip(*chunk)
        intr = intr0
        mix = torch.from_numpy(np.stack([np_complex_to_2ch(x) for x in mix_np])).to(device)
        cls = torch.full((len(chunk),), INTERFERENCE_TO_ID[intr], dtype=torch.long, device=device)
        with torch.no_grad():
            coarse_pred = coarse(mix, cls).float()
            if diffusion is not None:
                pred = sample_residual_ensemble(
                    diffusion,
                    mix,
                    coarse_pred,
                    cls,
                    schedule,
                    scales,
                    num_samples=posterior_samples,
                    steps=solver_steps,
                    sampler=sampler,
                    blend=blend,
                    seed=777000 + pos + INTERFERENCE_TO_ID[intr] * 100000,
                )
            else:
                pred = coarse_pred
        pred_np = twoch_to_np_complex(pred.detach().cpu().numpy())
        for j, idx in enumerate(ids):
            s_hat = pred_np[j]
            b_hat = mix_np[j] - s_hat
            b_true = mix_np[j] - target_np[j]
            rows.append(
                {
                    "method": method,
                    "interference": intr,
                    "idx": int(idx),
                    "sinr_db": infer_sinr_label(int(idx)),
                    "mse_soi": mse_complex_np(s_hat, target_np[j]),
                    "nmse_soi_db": nmse_db_complex_np(s_hat, target_np[j]),
                    "mse_interference": mse_complex_np(b_hat, b_true),
                }
            )
        pos = end
        pd.DataFrame(rows).sort_values(["interference", "idx"]).to_csv(csv_path, index=False)
        print(f"processed {pos}/{len(examples)}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No evaluation examples were processed")
    summary = (
        df.groupby(["interference", "sinr_db"], as_index=False)
        .agg(mse_soi=("mse_soi", "mean"), nmse_soi_db=("nmse_soi_db", "mean"), n=("idx", "count"))
        .sort_values(["interference", "sinr_db"], ascending=[True, False])
    )
    summary.to_csv(outdir / "summary_by_sinr.csv", index=False)
    overall = {
        "method": method,
        "examples": int(len(df)),
        "mean_mse_soi": float(df["mse_soi"].mean()),
        "mean_nmse_soi_db": float(df["nmse_soi_db"].mean()),
        "mean_mse_interference": float(df["mse_interference"].mean()),
        "blend": float(blend) if diffusion is not None else None,
        "sampler": sampler,
        "solver_steps": solver_steps,
        "posterior_samples": posterior_samples,
    }
    (outdir / "summary.json").write_text(json.dumps(overall, indent=2))
    print(json.dumps(overall, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
