#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from opdps_rf_sc.config import load_yaml
from opdps_rf_sc.data import MixtureBatchGenerator
from opdps_rf_sc.ema import EMA
from opdps_rf_sc.model_io import load_coarse_from_checkpoint
from opdps_rf_sc.models import build_separator
from opdps_rf_sc.rectified_flow import (
    flow_velocity,
    gaussian_kl_to_standard,
    sample_mixed_source,
    sample_rectified_flow,
)
from opdps_rf_sc.train_utils import (
    autocast_context,
    configure_torch,
    count_parameters,
    make_lr_scheduler,
    resolve_device,
    seed_all,
)


def load_scales(path, device):
    obj = json.loads(Path(path).read_text())

    return torch.tensor(
        obj["scales"],
        dtype=torch.float32,
        device=device,
    )


def complex_mse_ri(pred, target):
    """
    Match complex MSE:
        mean_t [(Ihat-I)^2 + (Qhat-Q)^2]
    """
    return (
        (pred - target)
        .square()
        .sum(dim=1)
        .mean(dim=1)
    )


def save_flow_checkpoint(
    path,
    *,
    flow,
    flow_ema,
    source,
    source_ema,
    optimizer,
    scheduler,
    step,
    config,
    mode,
    extra,
):
    obj = {
        "step": int(step),
        "mode": str(mode),
        "config": config,

        "flow_model": flow.state_dict(),
        "flow_ema": flow_ema.shadow.state_dict(),

        "source_model": (
            source.state_dict()
            if source is not None
            else None
        ),

        "source_ema": (
            source_ema.shadow.state_dict()
            if source_ema is not None
            else None
        ),

        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "extra": extra,
    }

    torch.save(obj, path)


@torch.no_grad()
def validate_reconstruction(
    *,
    flow,
    source,
    coarse,
    gen,
    scales,
    batches,
    batch_size,
    device,
    amp,
    mode,
    solver,
    ode_steps,
    source_weight,
    logvar_min,
    logvar_max,
    time_scale,
):
    flow.eval()

    if source is not None:
        source.eval()

    mse_vals = []

    for bi in range(int(batches)):
        batch = gen.batch(batch_size, device)

        with autocast_context(device, amp):
            coarse_pred = coarse(
                batch.mixture,
                batch.class_id,
            )

        coarse_pred = coarse_pred.float()

        generator = torch.Generator(
            device=device
        )
        generator.manual_seed(900000 + bi)

        residual_pred = sample_rectified_flow(
            flow,
            batch.mixture,
            coarse_pred,
            batch.class_id,
            mode=mode,
            source_model=source,
            source_weight=source_weight,
            solver=solver,
            steps=ode_steps,
            generator=generator,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
            time_scale=time_scale,
        )

        scale = scales[
            batch.class_id
        ].reshape(-1, 1, 1)

        pred = coarse_pred + scale * residual_pred

        mse_vals.extend(
            complex_mse_ri(
                pred.float(),
                batch.target.float(),
            )
            .detach()
            .cpu()
            .tolist()
        )

    flow.train()

    if source is not None:
        source.train()

    return sum(mse_vals) / max(
        1,
        len(mse_vals),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )
    ap.add_argument(
        "--cache-dir",
        required=True,
    )
    ap.add_argument(
        "--coarse-checkpoint",
        required=True,
    )
    ap.add_argument(
        "--residual-stats",
        required=True,
    )
    ap.add_argument(
        "--run-dir",
        required=True,
    )

    ap.add_argument(
        "--mode",
        choices=["gaussian", "mixed"],
        required=True,
    )

    ap.add_argument(
        "--device",
        default="auto",
    )

    ap.add_argument(
        "--steps",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    args = ap.parse_args()

    cfg = load_yaml(args.config)

    if args.seed is not None:
        cfg = dict(cfg)
        cfg["seed"] = int(args.seed)

    fcfg = dict(cfg["flow"])

    if args.steps is not None:
        fcfg["steps"] = int(args.steps)

    cfg = dict(cfg)
    cfg["flow"] = fcfg

    scfg = cfg["source"]

    run_dir = Path(
        args.run_dir
    ).expanduser().resolve()

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        run_dir
        / "resolved_config.json"
    ).write_text(
        json.dumps(
            cfg,
            indent=2,
        )
    )

    seed_all(
        int(cfg.get("seed", 1337))
        + (
            301
            if args.mode == "mixed"
            else 201
        )
    )

    configure_torch()

    device = resolve_device(
        args.device
    )

    amp = bool(
        fcfg.get(
            "amp_bf16",
            False,
        )
    )

    coarse, _ = (
        load_coarse_from_checkpoint(
            args.coarse_checkpoint,
            device,
            use_ema=True,
        )
    )

    for p in coarse.parameters():
        p.requires_grad_(False)

    flow = build_separator(
        fcfg["model"],
        in_channels=6,
        out_channels=2,
        time_conditioned=True,
    ).to(device)

    flow_ema = EMA(
        flow,
        decay=float(
            fcfg.get(
                "ema_decay",
                0.9999,
            )
        ),
    )

    source = None
    source_ema = None

    if args.mode == "mixed":
        source = build_separator(
            scfg["model"],
            in_channels=4,
            out_channels=4,
            time_conditioned=False,
        ).to(device)

        source_ema = EMA(
            source,
            decay=float(
                fcfg.get(
                    "ema_decay",
                    0.9999,
                )
            ),
        )

    params = list(
        flow.parameters()
    )

    if source is not None:
        params += list(
            source.parameters()
        )

    optimizer = torch.optim.AdamW(
        params,
        lr=float(fcfg["lr"]),
        weight_decay=float(
            fcfg.get(
                "weight_decay",
                1e-4,
            )
        ),
        betas=(0.9, 0.99),
    )

    total_steps = int(
        fcfg["steps"]
    )

    lr_scheduler = make_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(
            fcfg.get(
                "warmup_steps",
                0,
            )
        ),
        min_ratio=float(
            fcfg.get(
                "min_lr_ratio",
                0.05,
            )
        ),
    )

    scales = load_scales(
        args.residual_stats,
        device,
    )

    tcfg = cfg["train"]

    common_data = dict(
        cache_dir=args.cache_dir,

        length=int(
            tcfg.get(
                "length",
                40960,
            )
        ),

        continuous_sinr_prob=float(
            tcfg.get(
                "continuous_sinr_prob",
                0.2,
            )
        ),

        continuous_sinr_range=(
            float(
                tcfg.get(
                    "continuous_sinr_min",
                    -15,
                )
            ),
            float(
                tcfg.get(
                    "continuous_sinr_max",
                    5,
                )
            ),
        ),

        low_sinr_bias=float(
            tcfg.get(
                "low_sinr_bias",
                0.35,
            )
        ),

        common_phase_aug_prob=float(
            tcfg.get(
                "common_phase_aug_prob",
                0.5,
            )
        ),
    )

    train_gen = MixtureBatchGenerator(
        **common_data,
        split="train",
        seed=int(
            cfg.get(
                "seed",
                1337,
            )
        )
        + 311,
    )

    val_data = dict(
        common_data
    )

    val_data[
        "common_phase_aug_prob"
    ] = 0.0

    val_gen = MixtureBatchGenerator(
        **val_data,
        split="holdout",
        seed=616161,
    )

    batch_size = int(
        fcfg["batch_size"]
    )

    kl_weight = float(
        scfg.get(
            "kl_weight",
            1e-5,
        )
    )

    logvar_min = float(
        scfg.get(
            "logvar_min",
            -6.0,
        )
    )

    logvar_max = float(
        scfg.get(
            "logvar_max",
            3.0,
        )
    )

    time_scale = float(
        fcfg.get(
            "time_scale",
            1000.0,
        )
    )

    solver = str(
        fcfg.get(
            "val_solver",
            "heun",
        )
    )

    ode_steps = int(
        fcfg.get(
            "val_ode_steps",
            4,
        )
    )

    source_weight = float(
        fcfg.get(
            "val_source_weight",
            1.0,
        )
    )

    print(
        f"device={device}",
        f"mode={args.mode}",
        f"flow_params={count_parameters(flow):,}",
        f"source_params={count_parameters(source) if source is not None else 0:,}",
        f"scales={scales.tolist()}",
        flush=True,
    )

    best = float("inf")
    last = time.time()

    flow.train()

    if source is not None:
        source.train()

    for step in range(
        1,
        total_steps + 1,
    ):
        batch = train_gen.batch(
            batch_size,
            device,
        )

        with torch.no_grad(), autocast_context(
            device,
            amp,
        ):
            coarse_pred = coarse(
                batch.mixture,
                batch.class_id,
            )

        coarse_pred = (
            coarse_pred.float()
        )

        scale = scales[
            batch.class_id
        ].reshape(-1, 1, 1)

        # Normalized residual target.
        r1 = (
            batch.target
            - coarse_pred
        ) / scale

        if args.mode == "gaussian":
            r0 = torch.randn_like(
                r1
            )
            kl_loss = (
                r1.new_zeros(())
            )

        else:
            w = torch.rand(
                batch_size,
                device=device,
                dtype=r1.dtype,
            )

            r0, mean_w, var_w = (
                sample_mixed_source(
                    source,
                    batch.mixture,
                    coarse_pred,
                    batch.class_id,
                    weight=w,
                    logvar_min=logvar_min,
                    logvar_max=logvar_max,
                )
            )

            kl_loss = (
                gaussian_kl_to_standard(
                    mean_w,
                    var_w,
                )
            )

        t = torch.rand(
            batch_size,
            device=device,
            dtype=torch.float32,
        )

        tv = t.reshape(
            -1,
            1,
            1,
        )

        rt = (
            (1.0 - tv) * r0
            + tv * r1
        )

        velocity_target = (
            r1 - r0
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast_context(
            device,
            amp,
        ):
            velocity_pred = (
                flow_velocity(
                    flow,
                    rt,
                    batch.mixture,
                    coarse_pred,
                    batch.class_id,
                    t,
                    time_scale=time_scale,
                )
            )

            flow_loss = F.mse_loss(
                velocity_pred,
                velocity_target,
            )

            loss = (
                flow_loss
                + kl_weight
                * kl_loss
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            params,
            float(
                fcfg.get(
                    "grad_clip",
                    1.0,
                )
            ),
        )

        optimizer.step()
        lr_scheduler.step()

        flow_ema.update(flow)

        if source_ema is not None:
            source_ema.update(source)

        if (
            step % 100 == 0
            or step == 1
        ):
            dt = (
                time.time()
                - last
            )
            last = time.time()

            print(
                f"step={step:07d} "
                f"loss={float(loss.detach()):.6g} "
                f"flow={float(flow_loss.detach()):.6g} "
                f"kl={float(kl_loss.detach()):.6g} "
                f"lr={lr_scheduler.get_last_lr()[0]:.3e} "
                f"sec/100={dt:.2f}",
                flush=True,
            )

        validate_every = int(
            fcfg.get(
                "validate_every",
                2000,
            )
        )

        if (
            step % validate_every == 0
            or step == total_steps
        ):
            val_mse = (
                validate_reconstruction(
                    flow=flow_ema.shadow,
                    source=(
                        source_ema.shadow
                        if source_ema
                        is not None
                        else None
                    ),
                    coarse=coarse,
                    gen=val_gen,
                    scales=scales,
                    batches=int(
                        fcfg.get(
                            "val_batches",
                            4,
                        )
                    ),
                    batch_size=batch_size,
                    device=device,
                    amp=amp,
                    mode=args.mode,
                    solver=solver,
                    ode_steps=ode_steps,
                    source_weight=source_weight,
                    logvar_min=logvar_min,
                    logvar_max=logvar_max,
                    time_scale=time_scale,
                )
            )

            print(
                f"HOLDOUT "
                f"reconstruction_mse="
                f"{val_mse:.8g} "
                f"at step={step}",
                flush=True,
            )

            extra = {
                "best_val_mse": min(
                    best,
                    val_mse,
                ),
                "val_mse": val_mse,
                "coarse_checkpoint": str(
                    Path(
                        args.coarse_checkpoint
                    ).resolve()
                ),
                "residual_stats": str(
                    Path(
                        args.residual_stats
                    ).resolve()
                ),
            }

            save_flow_checkpoint(
                run_dir / "latest.pt",
                flow=flow,
                flow_ema=flow_ema,
                source=source,
                source_ema=source_ema,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                step=step,
                config=cfg,
                mode=args.mode,
                extra=extra,
            )

            if val_mse < best:
                best = val_mse

                extra[
                    "best_val_mse"
                ] = best

                save_flow_checkpoint(
                    run_dir / "best.pt",
                    flow=flow,
                    flow_ema=flow_ema,
                    source=source,
                    source_ema=source_ema,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    step=step,
                    config=cfg,
                    mode=args.mode,
                    extra=extra,
                )

        elif (
            step
            % int(
                fcfg.get(
                    "save_every",
                    5000,
                )
            )
            == 0
        ):
            save_flow_checkpoint(
                run_dir / "latest.pt",
                flow=flow,
                flow_ema=flow_ema,
                source=source,
                source_ema=source_ema,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                step=step,
                config=cfg,
                mode=args.mode,
                extra={
                    "best_val_mse":
                    best
                },
            )


if __name__ == "__main__":
    main()
