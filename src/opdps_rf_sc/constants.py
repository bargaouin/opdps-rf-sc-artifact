from __future__ import annotations

SIGNAL_TYPES = ("EMISignal1", "CommSignal2", "CommSignal3")
SEPARATION_INTERFERERS = ("EMISignal1", "CommSignal3")
INTERFERENCE_TO_ID = {name: i for i, name in enumerate(SEPARATION_INTERFERERS)}
ID_TO_INTERFERENCE = {i: name for name, i in INTERFERENCE_TO_ID.items()}

# These counts are shown by the official QuickStart notebook for the June 2021 dataset.
OFFICIAL_TRAIN_FRAME_COUNTS = {
    "EMISignal1": 530,
    "CommSignal2": 100,
    "CommSignal3": 139,
}

# The validation/test generation uses 11 SINR levels from +3 to -12 dB in 1.5 dB steps.
OFFICIAL_SINR_LEVELS_DESC = (3.0, 1.5, 0.0, -1.5, -3.0, -4.5, -6.0, -7.5, -9.0, -10.5, -12.0)
OFFICIAL_SINR_LEVELS_ASC = tuple(reversed(OFFICIAL_SINR_LEVELS_DESC))
OFFICIAL_MIXTURE_LENGTH = 40960
OFFICIAL_SAMPLE_RATE = 25_000_000
