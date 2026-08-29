from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


class EMALossNormalizer(nn.Module):
    """EMA scale tracker for heterogeneous tactical losses.

    Tactical preferences remain encoded by explicit semantic weights.  This
    module only estimates numerical scales.  EMA state is stored as registered
    buffers so it is checkpointable and moves with ``.to(device)``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        beta: float = 0.99,
        eps: float = 1e-6,
        clip_min: float = 0.1,
        clip_max: float = 10.0,
    ):
        super().__init__()
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if clip_min <= 0.0 or clip_max < clip_min:
            raise ValueError("invalid clipping interval")

        self.keys = tuple(str(key) for key in keys)
        if not self.keys:
            raise ValueError("at least one loss key is required")

        self.beta = float(beta)
        self.eps = float(eps)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        for key in self.keys:
            safe = self._safe_name(key)
            self.register_buffer(f"ema_{safe}", torch.tensor(0.0))
            self.register_buffer(
                f"initialized_{safe}",
                torch.tensor(False, dtype=torch.bool),
            )

    @staticmethod
    def _safe_name(key: str) -> str:
        return key.replace("-", "_").replace("/", "_")

    def _ema_buffer(self, key: str) -> Tensor:
        if key not in self.keys:
            raise KeyError(f"unknown EMA loss key: {key}")
        return getattr(self, f"ema_{self._safe_name(key)}")

    def _initialized_buffer(self, key: str) -> Tensor:
        if key not in self.keys:
            raise KeyError(f"unknown EMA loss key: {key}")
        return getattr(self, f"initialized_{self._safe_name(key)}")

    @torch.no_grad()
    def update(self, losses: Mapping[str, Tensor]) -> None:
        """Update EMA scales from detached batch losses."""
        for key in self.keys:
            if key not in losses:
                continue

            value = losses[key].detach().float()
            if value.numel() != 1:
                value = value.mean()
            if not bool(torch.isfinite(value)):
                continue

            value = value.clamp_min(self.eps)
            ema = self._ema_buffer(key)
            initialized = self._initialized_buffer(key)

            if not bool(initialized):
                ema.copy_(value)
                initialized.fill_(True)
            else:
                ema.mul_(self.beta).add_(value * (1.0 - self.beta))

    def scale(self, key: str, like: Tensor | None = None) -> Tensor:
        ema = self._ema_buffer(key)
        initialized = self._initialized_buffer(key)
        if bool(initialized):
            result = ema.detach().clamp_min(self.eps)
        else:
            result = torch.ones_like(ema)

        if like is not None:
            result = result.to(device=like.device, dtype=like.dtype)
        return result

    def normalize(self, key: str, loss: Tensor) -> Tensor:
        value = loss / self.scale(key, like=loss)
        return torch.clamp(value, self.clip_min, self.clip_max)

    def snapshot(self) -> dict[str, float]:
        return {
            key: float(self.scale(key).detach().cpu())
            for key in self.keys
        }

    def initialized(self) -> dict[str, bool]:
        return {
            key: bool(self._initialized_buffer(key).detach().cpu())
            for key in self.keys
        }
