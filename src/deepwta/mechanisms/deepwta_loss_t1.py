from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter

from deepwta.mechanisms.conflict_regularizer import FrozenConflictRegularizer
from deepwta.training.loss_normalizer import EMALossNormalizer


TACTICAL_EMA_KEYS = (
    "ammo",
    "crossing",
    "distance",
    "interception",
)


class DeepWTALossT1(nn.Module):
    """DeepWTA T1 loss: fixed semantics + EMA numerical scale normalization."""

    def __init__(
        self,
        num_weapons: int,
        num_schemes: int,
        inventory_max: float,
        conflict_regularizer: FrozenConflictRegularizer | None = None,
        demand_tolerance: float = 0.05,
        normalizer: EMALossNormalizer | None = None,
    ):
        super().__init__()
        self.num_weapons = int(num_weapons)
        self.num_schemes = int(num_schemes)
        self.inventory_max = float(inventory_max)
        self.conflict_regularizer = conflict_regularizer
        self.demand_tolerance = float(demand_tolerance)
        self.normalizer = normalizer

    @staticmethod
    def _weight(weights: Mapping[str, float], key: str) -> float:
        if key not in weights:
            raise KeyError(f"missing loss weight: {key}")
        return float(weights[key])

    def forward(
        self,
        batch,
        output,
        weights: Mapping[str, float],
        objective_scale: float = 1.0,
        conflict_sampling_step: int = 0,
        loss_mode: str = "raw",
        ema_strength: float = 1.0,
        update_ema: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if loss_mode not in {"raw", "ema"}:
            raise ValueError(f"unknown loss_mode: {loss_mode}")
        if loss_mode == "ema" and self.normalizer is None:
            raise RuntimeError("EMA loss mode requested without a normalizer")

        k = self.num_weapons
        device = output.allocation.device

        allocation = output.allocation.float()
        allocation_pre = output.allocation_pre_projection.float()
        target_demand = output.target_demand.float()
        inventory_scale = output.inventory_scale.float()
        scheme_probs_raw = output.scheme_probs.float()
        dtype = allocation.dtype

        src = batch.edge_index[0]
        edge_attr = batch.edge_attr.float()
        edge_q = edge_attr[:, 1 : 1 + k]
        distance_norm = edge_attr[:, 0].clamp_min(0.0)

        effective_edge = allocation * (1.0 - edge_q)
        effective_target = scatter(
            effective_edge,
            output.edge_target_index,
            dim=0,
            dim_size=target_demand.size(0),
            reduce="sum",
        )
        shortage = (target_demand - effective_target).clamp_min(0.0)
        total_required = target_demand.sum().clamp_min(1.0)
        demand_loss = shortage.sum() / total_required

        active_demand = target_demand > 1e-6
        if bool(active_demand.any()):
            demand_satisfaction_rate = (
                shortage[active_demand] <= self.demand_tolerance
            ).float().mean()
        else:
            demand_satisfaction_rate = torch.ones(
                (), dtype=dtype, device=device
            )

        num_nodes = batch.x.size(0)
        inventory = batch.x[:, 3 : 3 + k] * self.inventory_max
        base_mask = batch.x[:, 2] < 0.5

        used_pre = scatter(
            allocation_pre,
            src,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        overload_pre = (used_pre - inventory).clamp_min(0.0)
        inventory_denominator = inventory[base_mask].sum().clamp_min(1.0)
        inventory_pre_loss = (
            overload_pre[base_mask].sum() / inventory_denominator
        )

        used_post = scatter(
            allocation,
            src,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        overload_post = (used_post - inventory).clamp_min(0.0)
        inventory_post_violation = (
            overload_post[base_mask].sum() / inventory_denominator
        )
        projection_pressure = (
            1.0 - inventory_scale[base_mask]
        ).clamp_min(0.0).mean()

        total_launches = allocation.sum()
        ammo_loss = total_launches / total_required

        allocation_denominator = total_launches.clamp_min(1e-6)
        interception_loss = (
            allocation * edge_q
        ).sum() / allocation_denominator
        distance_loss = (
            allocation * distance_norm.unsqueeze(-1)
        ).sum() / allocation_denominator

        scheme_probs = scheme_probs_raw.clamp_min(1e-8)
        scheme_entropy = -(
            scheme_probs * scheme_probs.log()
        ).sum(dim=-1)
        scheme_entropy = (
            scheme_entropy / max(math.log(self.num_schemes), 1e-8)
        ).mean()

        crossing_weight = self._weight(weights, "crossing")
        if (
            self.conflict_regularizer is not None
            and crossing_weight > 0.0
            and objective_scale > 0.0
        ):
            conflict = self.conflict_regularizer(
                batch,
                output.route_load,
                sampling_step=conflict_sampling_step,
            )
            crossing_loss = conflict.loss
            conflict_expected_count = conflict.mean_expected_count
            conflict_propensity = conflict.mean_propensity
            route_activation = conflict.mean_route_activation
            route_coverage = conflict.mean_route_coverage
            conflict_candidate_pairs = conflict.mean_candidate_pairs
        else:
            crossing_loss = output.route_load.sum() * 0.0
            zero = crossing_loss.detach()
            conflict_expected_count = zero
            conflict_propensity = zero
            route_activation = zero
            route_coverage = zero
            conflict_candidate_pairs = zero

        tactical_raw = {
            "ammo": ammo_loss,
            "crossing": crossing_loss,
            "distance": distance_loss,
            "interception": interception_loss,
        }

        if (
            update_ema
            and self.normalizer is not None
            and self.training
            and objective_scale > 0.0
        ):
            self.normalizer.update(tactical_raw)

        ema_strength = float(max(0.0, min(1.0, ema_strength)))
        tactical_effective: dict[str, Tensor] = {}
        tactical_normalized: dict[str, Tensor] = {}

        for key, raw_loss in tactical_raw.items():
            crossing_disabled = (
                key == "crossing"
                and (
                    self.conflict_regularizer is None
                    or crossing_weight <= 0.0
                    or objective_scale <= 0.0
                )
            )
            if self.normalizer is not None and not crossing_disabled:
                normalized = self.normalizer.normalize(key, raw_loss)
            else:
                normalized = raw_loss
            tactical_normalized[key] = normalized

            if loss_mode == "ema":
                tactical_effective[key] = (
                    (1.0 - ema_strength) * raw_loss
                    + ema_strength * normalized
                )
            else:
                tactical_effective[key] = raw_loss

        objective_scale = float(max(0.0, min(1.0, objective_scale)))
        hard_total = (
            self._weight(weights, "demand") * demand_loss
            + self._weight(weights, "inventory_pre") * inventory_pre_loss
        )
        raw_soft_total = (
            self._weight(weights, "scheme_entropy") * scheme_entropy
            + self._weight(weights, "ammo") * ammo_loss
            + self._weight(weights, "interception") * interception_loss
            + self._weight(weights, "distance") * distance_loss
            + crossing_weight * crossing_loss
        )
        effective_soft_total = (
            self._weight(weights, "scheme_entropy") * scheme_entropy
            + self._weight(weights, "ammo") * tactical_effective["ammo"]
            + self._weight(weights, "interception")
            * tactical_effective["interception"]
            + self._weight(weights, "distance")
            * tactical_effective["distance"]
            + crossing_weight * tactical_effective["crossing"]
        )

        total = hard_total + objective_scale * effective_soft_total
        raw_reference_loss = hard_total + objective_scale * raw_soft_total

        stats = {
            "loss": total,
            "raw_reference_loss": raw_reference_loss,
            "hard_total": hard_total,
            "raw_soft_total": raw_soft_total,
            "effective_soft_total": effective_soft_total,
            "ema_strength": torch.tensor(
                ema_strength, dtype=dtype, device=device
            ),
            "demand": demand_loss,
            "inventory_pre": inventory_pre_loss,
            "inventory_post": inventory_post_violation,
            "projection_pressure": projection_pressure,
            "scheme_entropy": scheme_entropy,
            "ammo": ammo_loss,
            "interception": interception_loss,
            "distance": distance_loss,
            "crossing": crossing_loss,
            "norm_ammo": tactical_normalized["ammo"],
            "norm_crossing": tactical_normalized["crossing"],
            "norm_distance": tactical_normalized["distance"],
            "norm_interception": tactical_normalized["interception"],
            "demand_satisfaction_rate": demand_satisfaction_rate,
            "mean_shortage": shortage.mean(),
            "max_shortage": shortage.max() if shortage.numel() else torch.zeros(
                (), dtype=dtype, device=device
            ),
            "conflict_expected_count": conflict_expected_count,
            "conflict_propensity": conflict_propensity,
            "route_activation": route_activation,
            "route_coverage": route_coverage,
            "conflict_candidate_pairs": conflict_candidate_pairs,
        }

        if self.normalizer is not None:
            for key in TACTICAL_EMA_KEYS:
                stats[f"ema_scale_{key}"] = self.normalizer.scale(
                    key, like=allocation
                )

        return total, stats
