import torch

from opdps_rf_sc.diffusion import DiffusionSchedule, sample_ddim
from opdps_rf_sc.models import build_separator


def test_schedule_and_ddim_shapes():
    cfg = {
        "kind": "hybrid_operator", "base_channels": 8, "channel_mults": [1, 2, 4],
        "emb_dim": 32, "operator_blocks": 1, "max_modes": 8, "spectral_rank": 4,
        "residual_to_input": False,
    }
    model = build_separator(cfg, in_channels=6, out_channels=2, time_conditioned=True)
    schedule = DiffusionSchedule(20)
    mix = torch.randn(2, 2, 128)
    coarse = torch.randn(2, 2, 128)
    cls = torch.tensor([0, 1])
    out = sample_ddim(model, mix, coarse, cls, schedule, steps=4)
    assert out.shape == coarse.shape
    assert torch.isfinite(out).all()
