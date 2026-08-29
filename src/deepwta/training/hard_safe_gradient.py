from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class GradientSurgeryStats:
    cosine: float
    conflict: float
    hard_grad_norm: float
    soft_grad_norm: float
    projection_coefficient: float
    projected_soft_fraction: float

    def as_dict(self) -> dict[str, float]:
        return {
            "hard_soft_grad_cosine": self.cosine,
            "hard_soft_grad_conflict": self.conflict,
            "hard_grad_norm": self.hard_grad_norm,
            "soft_grad_norm": self.soft_grad_norm,
            "projection_coefficient": self.projection_coefficient,
            "projected_soft_fraction": self.projected_soft_fraction,
        }


def _zeros_like_or_none(
    grad: Tensor | None,
    param: Tensor,
) -> Tensor:
    if grad is None:
        return torch.zeros_like(param)
    return grad


def hard_safe_projected_gradients(
    parameters: Iterable[torch.nn.Parameter],
    hard_loss: Tensor,
    soft_loss: Tensor,
    *,
    accumulation_scale: float = 1.0,
    eps: float = 1e-12,
) -> GradientSurgeryStats:
    """Accumulate hard-protected multi-objective gradients.

    If <g_h, g_s> < 0, project only the soft gradient onto the orthogonal
    complement of g_h:

        g_s' = g_s - <g_s,g_h>/||g_h||^2 * g_h

    and optimize with:

        g = g_h + g_s'

    This is deliberately asymmetric: hard feasibility is protected and the
    tactical gradient is modified only when it conflicts with the hard
    constraint gradient.

    Gradients are accumulated into ``param.grad`` so normal gradient
    accumulation can still be used.
    """
    params = [p for p in parameters if p.requires_grad]
    if not params:
        return GradientSurgeryStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    scale = float(accumulation_scale)
    if scale <= 0:
        raise ValueError("accumulation_scale must be positive")

    g_h_raw = torch.autograd.grad(
        hard_loss / scale,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    g_s_raw = torch.autograd.grad(
        soft_loss / scale,
        params,
        retain_graph=False,
        allow_unused=True,
    )

    g_h = [
        _zeros_like_or_none(g, p)
        for g, p in zip(g_h_raw, params)
    ]
    g_s = [
        _zeros_like_or_none(g, p)
        for g, p in zip(g_s_raw, params)
    ]

    device = params[0].device
    dot = torch.zeros((), device=device, dtype=torch.float32)
    hard_sq = torch.zeros((), device=device, dtype=torch.float32)
    soft_sq = torch.zeros((), device=device, dtype=torch.float32)

    for gh, gs in zip(g_h, g_s):
        ghf = gh.float()
        gsf = gs.float()
        dot = dot + torch.sum(ghf * gsf)
        hard_sq = hard_sq + torch.sum(ghf * ghf)
        soft_sq = soft_sq + torch.sum(gsf * gsf)

    hard_norm = torch.sqrt(hard_sq.clamp_min(0.0))
    soft_norm = torch.sqrt(soft_sq.clamp_min(0.0))
    cosine = dot / (hard_norm * soft_norm + eps)

    conflict = bool(
        dot.detach().item() < 0.0
        and hard_sq.detach().item() > eps
    )

    if conflict:
        # dot is negative, so coefficient is negative and subtracting it adds
        # the component needed to remove the conflicting projection.
        coefficient = dot / (hard_sq + eps)
    else:
        coefficient = torch.zeros_like(dot)

    projected_sq = torch.zeros_like(soft_sq)
    for p, gh, gs in zip(params, g_h, g_s):
        if conflict:
            gs_projected = gs - coefficient.to(gs.dtype) * gh
            removed = gs - gs_projected
            projected_sq = projected_sq + torch.sum(
                removed.float() * removed.float()
            )
        else:
            gs_projected = gs

        combined = gh + gs_projected
        if p.grad is None:
            p.grad = combined.detach().clone()
        else:
            p.grad.add_(combined.detach())

    projected_fraction = torch.sqrt(projected_sq.clamp_min(0.0)) / (
        soft_norm + eps
    )

    return GradientSurgeryStats(
        cosine=float(cosine.detach().cpu()),
        conflict=float(conflict),
        hard_grad_norm=float(hard_norm.detach().cpu()),
        soft_grad_norm=float(soft_norm.detach().cpu()),
        projection_coefficient=float(coefficient.detach().cpu()),
        projected_soft_fraction=float(projected_fraction.detach().cpu()),
    )
