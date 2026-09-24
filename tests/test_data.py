import numpy as np

from opdps_rf_sc.data import MixtureBatchGenerator
from opdps_rf_sc.complex_utils import twoch_to_np_complex, measured_sinr_db_np


def test_mixture_generator_additivity_and_sinr(tiny_cache):
    gen = MixtureBatchGenerator(
        str(tiny_cache), split="train", length=256,
        sinr_levels=[-6.0], continuous_sinr_prob=0.0, low_sinr_bias=0.0,
        common_phase_aug_prob=0.0, seed=10,
    )
    mix, target, intr, cls, sinr = gen.sample_numpy(4)
    y = twoch_to_np_complex(mix)
    s = twoch_to_np_complex(target)
    b = twoch_to_np_complex(intr)
    assert np.allclose(y, s + b, atol=1e-5)
    assert np.allclose(np.mean(np.abs(s) ** 2, axis=-1), 1.0, atol=1e-5)
    for k in range(4):
        assert abs(measured_sinr_db_np(s[k], b[k]) + 6.0) < 1e-3
    assert set(cls.tolist()).issubset({0, 1})
