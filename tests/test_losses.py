import torch

from opdps_rf_sc.losses import separation_loss


def test_loss_zero_on_identity():
    x = torch.randn(2, 2, 128)
    loss, parts = separation_loss(x, x, {"l1_weight": 0.05, "spectral_weight": 0.02})
    assert float(loss) < 1e-6
    assert parts["mse"] == 0.0
