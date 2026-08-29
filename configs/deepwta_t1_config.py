from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepWTAT1Config:
    """Step-1 training-only optimization.

    T1 deliberately changes only the numerical scaling of the existing
    tactical objectives.  No model architecture, repair logic, dataset, hard
    constraint weight, or tactical semantic weight is changed.
    """

    ema_keys: tuple[str, ...] = (
        "ammo",
        "crossing",
        "distance",
        "interception",
    )
    ema_beta: float = 0.99
    ema_eps: float = 1e-6
    ema_clip_min: float = 0.1
    ema_clip_max: float = 10.0

    # BALANCED collects EMA statistics while still optimizing the original raw
    # loss.  FULL blends raw -> EMA-normalized objectives over this many epochs.
    full_normalization_ramp_epochs: int = 15


T1_CONFIG = DeepWTAT1Config()
