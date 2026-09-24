from __future__ import annotations

import torch


def source_parameters(
    source_model,
    mixture: torch.Tensor,
    coarse: torch.Tensor,
    class_id: torch.Tensor,
    *,
    logvar_min: float = -6.0,
    logvar_max: float = 3.0,
):
    """
    Predict q_phi(r0 | kappa), where
        kappa = [mixture, coarse estimate, class].

    Source network outputs:
        [mu_I, mu_Q, logvar_I, logvar_Q].
    """
    inp = torch.cat([mixture, coarse], dim=1)

    stats = source_model(inp, class_id)

    if stats.shape[1] != 4:
        raise RuntimeError(
            f"Source predictor must output 4 channels, got {stats.shape}"
        )

    mu, logvar = torch.chunk(stats, 2, dim=1)

    logvar = logvar.clamp(
        min=float(logvar_min),
        max=float(logvar_max),
    )

    return mu, logvar


def sample_mixed_source(
    source_model,
    mixture: torch.Tensor,
    coarse: torch.Tensor,
    class_id: torch.Tensor,
    *,
    weight,
    generator=None,
    logvar_min: float = -6.0,
    logvar_max: float = 3.0,
):
    """
    MixFlow-style Gaussian source:

        q(r0 | kappa,w)
        = N(w mu_kappa,
            w Sigma_kappa + (1-w) I)

    where Sigma is diagonal.
    """
    mu, logvar = source_parameters(
        source_model,
        mixture,
        coarse,
        class_id,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )

    var = logvar.exp()

    if not torch.is_tensor(weight):
        weight = torch.tensor(
            float(weight),
            device=mu.device,
            dtype=mu.dtype,
        )

    if weight.ndim == 0:
        weight = weight.reshape(1, 1, 1)

    if weight.ndim == 1:
        weight = weight.reshape(-1, 1, 1)

    mean_w = weight * mu
    var_w = weight * var + (1.0 - weight)

    eps = torch.randn(
        mean_w.shape,
        device=mean_w.device,
        dtype=mean_w.dtype,
        generator=generator,
    )

    r0 = mean_w + var_w.clamp_min(1e-8).sqrt() * eps

    return r0, mean_w, var_w


def gaussian_kl_to_standard(
    mean: torch.Tensor,
    var: torch.Tensor,
) -> torch.Tensor:
    """
    KL[N(mean,var) || N(0,I)] averaged over batch/channels/time.
    """
    return 0.5 * (
        mean.square()
        + var
        - torch.log(var.clamp_min(1e-8))
        - 1.0
    ).mean()


def flow_velocity(
    model,
    x: torch.Tensor,
    mixture: torch.Tensor,
    coarse: torch.Tensor,
    class_id: torch.Tensor,
    t: torch.Tensor,
    *,
    time_scale: float = 1000.0,
):
    """
    Conditional velocity:
        v_theta(x_t, t | mixture, coarse, class).
    """
    inp = torch.cat([x, mixture, coarse], dim=1)

    return model(
        inp,
        class_id,
        t=t.float() * float(time_scale),
    )


@torch.no_grad()
def sample_rectified_flow(
    model,
    mixture: torch.Tensor,
    coarse: torch.Tensor,
    class_id: torch.Tensor,
    *,
    mode: str,
    source_model=None,
    source_weight: float = 1.0,
    solver: str = "heun",
    steps: int = 4,
    generator=None,
    logvar_min: float = -6.0,
    logvar_max: float = 3.0,
    time_scale: float = 1000.0,
):
    """
    Integrate r from t=0 -> 1.

    Heun:
        steps=4 corresponds to approximately 8 vector-field evaluations.

    Euler:
        steps=8 corresponds to 8 vector-field evaluations.
    """
    if mode == "gaussian":
        x = torch.randn(
            (mixture.shape[0], 2, mixture.shape[-1]),
            device=mixture.device,
            dtype=mixture.dtype,
            generator=generator,
        )

    elif mode == "mixed":
        if source_model is None:
            raise ValueError("mixed mode requires source_model")

        x, _, _ = sample_mixed_source(
            source_model,
            mixture,
            coarse,
            class_id,
            weight=float(source_weight),
            generator=generator,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )

    else:
        raise ValueError(f"Unknown flow mode: {mode}")

    times = torch.linspace(
        0.0,
        1.0,
        int(steps) + 1,
        device=mixture.device,
        dtype=torch.float32,
    )

    b = mixture.shape[0]

    for i in range(int(steps)):
        t0 = times[i]
        t1 = times[i + 1]
        dt = t1 - t0

        tb0 = torch.full(
            (b,),
            float(t0),
            device=mixture.device,
        )

        v0 = flow_velocity(
            model,
            x,
            mixture,
            coarse,
            class_id,
            tb0,
            time_scale=time_scale,
        )

        if solver == "euler":
            x = x + dt * v0

        elif solver == "heun":
            x_euler = x + dt * v0

            tb1 = torch.full(
                (b,),
                float(t1),
                device=mixture.device,
            )

            v1 = flow_velocity(
                model,
                x_euler,
                mixture,
                coarse,
                class_id,
                tb1,
                time_scale=time_scale,
            )

            x = x + 0.5 * dt * (v0 + v1)

        else:
            raise ValueError(
                f"Unknown solver: {solver}"
            )

    return x
