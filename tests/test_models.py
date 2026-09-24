import torch

from opdps_rf_sc.models import build_separator


def tiny_cfg(kind="hybrid_operator"):
    if kind == "hybrid_operator":
        return {
            "kind": kind,
            "base_channels": 8,
            "channel_mults": [1, 2, 4],
            "emb_dim": 32,
            "operator_blocks": 2,
            "max_modes": 16,
            "spectral_rank": 4,
            "residual_to_input": True,
        }
    if kind == "waveunet":
        return {
            "kind": kind,
            "base_channels": 8,
            "channel_mults": [1, 2, 4],
            "emb_dim": 32,
            "residual_to_input": True,
        }
    return {
        "kind": "fno", "width": 8, "emb_dim": 32,
        "operator_blocks": 2, "max_modes": 16, "spectral_rank": 4,
    }


def test_length_agnostic_forward():
    for kind in ("hybrid_operator", "waveunet", "fno"):
        model = build_separator(tiny_cfg(kind))
        for length in (255, 256, 513):
            x = torch.randn(2, 2, length)
            c = torch.tensor([0, 1])
            y = model(x, c)
            assert y.shape == x.shape
            assert torch.isfinite(y).all()


def test_time_conditioned_diffusion_network():
    cfg = tiny_cfg("hybrid_operator")
    cfg["residual_to_input"] = False
    model = build_separator(cfg, in_channels=6, out_channels=2, time_conditioned=True)
    x = torch.randn(2, 6, 257)
    c = torch.tensor([0, 1])
    t = torch.tensor([1.0, 999.0])
    y = model(x, c, t=t)
    assert y.shape == (2, 2, 257)
