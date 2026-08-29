from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from deepwta.evaluation.ammo_lower_bound import (
    TargetSchemeBound,
    minimize_fractional_bound,
    minimum_raw_shot_bound,
    selected_scheme_bound,
)


@dataclass(frozen=True)
class RawBoundChoice:
    shots: int
    demand: float
    ammo_norm: float
    scheme_idx: np.ndarray


def _choice_from_scheme_idx(
    table: List[List[TargetSchemeBound]],
    scheme_idx: np.ndarray,
    shot_field: str,
) -> RawBoundChoice:
    scheme_idx = np.asarray(scheme_idx, dtype=np.int64).reshape(-1)
    if len(table) != scheme_idx.size:
        raise ValueError("scheme_idx length does not match target count")

    shots = 0
    demand = 0.0
    for target, scheme in enumerate(scheme_idx.tolist()):
        item = table[target][int(scheme)]
        value = getattr(item, shot_field)
        if not item.legal or not item.feasible or value is None:
            raise ValueError("chosen scheme is infeasible in raw-ammo audit")
        shots += int(value)
        demand += float(item.demand_total)

    return RawBoundChoice(
        shots=int(shots),
        demand=float(demand),
        ammo_norm=float(shots / max(demand, 1e-12)),
        scheme_idx=scheme_idx.copy(),
    )


def minimum_raw_choice(
    table: List[List[TargetSchemeBound]],
    shot_field: str = "capacity_shots",
) -> RawBoundChoice:
    shots, scheme_idx = minimum_raw_shot_bound(table, shot_field)
    result = _choice_from_scheme_idx(table, scheme_idx, shot_field)
    if result.shots != shots:
        raise AssertionError("raw-shot lower-bound accounting mismatch")
    return result


def _safe_share(value: float, total: float) -> float:
    if total <= 1e-12:
        return 0.0
    return float(max(value, 0.0) / total)


def audit_raw_ammo_scheme_gap(
    A: np.ndarray,
    scheme_idx: np.ndarray,
    table: List[List[TargetSchemeBound]],
    near_raw_lb_ratio: float = 0.02,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Decompose raw ammunition gap into allocation, scheme and inventory terms.

    Raw-shot decomposition (all optimistic lower bounds):

        actual_shots - global_unlimited_raw_lb
          = (actual - selected_scheme_capacity_lb)
          + (selected_scheme_capacity_lb - global_raw_capacity_lb)
          + (global_raw_capacity_lb - global_raw_unlimited_lb)

    The first term reflects route/allocation and cross-target inventory coupling,
    the second reflects scheme choice, and the third reflects local route
    capacity. Crossing is intentionally ignored by the lower bounds.
    """
    A = np.asarray(A, dtype=np.int64)
    scheme_idx = np.asarray(scheme_idx, dtype=np.int64).reshape(-1)
    actual_shots = int(np.sum(A))

    selected_cap_shots, selected_demand, selected_cap_norm = selected_scheme_bound(
        table, scheme_idx, "capacity_shots"
    )
    selected_unlim_shots, _, selected_unlim_norm = selected_scheme_bound(
        table, scheme_idx, "unlimited_shots"
    )

    raw_cap = minimum_raw_choice(table, "capacity_shots")
    raw_unlim = minimum_raw_choice(table, "unlimited_shots")
    norm_cap = minimize_fractional_bound(table, "capacity_shots")
    norm_unlim = minimize_fractional_bound(table, "unlimited_shots")

    allocation_gap = max(actual_shots - selected_cap_shots, 0)
    scheme_gap = max(selected_cap_shots - raw_cap.shots, 0)
    local_inventory_gap = max(raw_cap.shots - raw_unlim.shots, 0)
    total_gap = max(actual_shots - raw_unlim.shots, 0)

    raw_capacity_gap_ratio = float(
        actual_shots / max(float(raw_cap.shots), 1e-12) - 1.0
    )
    raw_unlimited_gap_ratio = float(
        actual_shots / max(float(raw_unlim.shots), 1e-12) - 1.0
    )

    components = {
        "allocation_cross_target_gap": float(allocation_gap),
        "scheme_choice_gap": float(scheme_gap),
        "local_inventory_gap": float(local_inventory_gap),
    }
    if raw_capacity_gap_ratio <= near_raw_lb_ratio or total_gap <= 0:
        bottleneck = "near_raw_relaxed_lb"
    else:
        name, value = max(components.items(), key=lambda item: item[1])
        bottleneck = (
            name
            if _safe_share(value, float(total_gap)) >= 0.60
            else "mixed"
        )

    raw_vs_norm_same = np.asarray(raw_cap.scheme_idx) == np.asarray(norm_cap.scheme_idx)
    selected_vs_raw = scheme_idx == np.asarray(raw_cap.scheme_idx)
    selected_vs_norm = scheme_idx == np.asarray(norm_cap.scheme_idx)

    scenario: Dict[str, object] = {
        "actual_raw_ammo": actual_shots,
        "actual_total_demand": float(selected_demand),
        "actual_ammo_norm": float(actual_shots / max(selected_demand, 1e-12)),
        "selected_scheme_capacity_lb_raw": int(selected_cap_shots),
        "selected_scheme_unlimited_lb_raw": int(selected_unlim_shots),
        "selected_scheme_capacity_lb_norm": float(selected_cap_norm),
        "selected_scheme_unlimited_lb_norm": float(selected_unlim_norm),
        "global_raw_capacity_lb_shots": int(raw_cap.shots),
        "global_raw_capacity_lb_demand": float(raw_cap.demand),
        "global_raw_capacity_lb_implied_norm": float(raw_cap.ammo_norm),
        "global_raw_unlimited_lb_shots": int(raw_unlim.shots),
        "global_raw_unlimited_lb_demand": float(raw_unlim.demand),
        "global_raw_unlimited_lb_implied_norm": float(raw_unlim.ammo_norm),
        "global_norm_capacity_lb_shots": int(norm_cap.total_shots),
        "global_norm_capacity_lb_demand": float(norm_cap.total_demand),
        "global_norm_capacity_lb_norm": float(norm_cap.ammo_norm),
        "global_norm_unlimited_lb_shots": int(norm_unlim.total_shots),
        "global_norm_unlimited_lb_demand": float(norm_unlim.total_demand),
        "global_norm_unlimited_lb_norm": float(norm_unlim.ammo_norm),
        "raw_allocation_cross_target_gap_shots": int(allocation_gap),
        "raw_scheme_choice_gap_shots": int(scheme_gap),
        "raw_local_inventory_gap_shots": int(local_inventory_gap),
        "raw_total_gap_to_unlimited_lb_shots": int(total_gap),
        "raw_allocation_gap_share": _safe_share(allocation_gap, float(total_gap)),
        "raw_scheme_gap_share": _safe_share(scheme_gap, float(total_gap)),
        "raw_local_inventory_gap_share": _safe_share(
            local_inventory_gap, float(total_gap)
        ),
        "raw_gap_to_capacity_lb_ratio": raw_capacity_gap_ratio,
        "raw_gap_to_unlimited_lb_ratio": raw_unlimited_gap_ratio,
        "near_raw_capacity_lb": bool(raw_capacity_gap_ratio <= near_raw_lb_ratio),
        "raw_bottleneck_class": bottleneck,
        "selected_vs_raw_opt_target_match_rate": float(np.mean(selected_vs_raw)),
        "selected_vs_norm_opt_target_match_rate": float(np.mean(selected_vs_norm)),
        "raw_opt_vs_norm_opt_target_match_rate": float(np.mean(raw_vs_norm_same)),
        "raw_opt_changed_target_count": int(np.sum(~selected_vs_raw)),
        "norm_opt_changed_target_count": int(np.sum(~selected_vs_norm)),
        "norm_opt_shot_premium_vs_raw_opt": int(norm_cap.total_shots - raw_cap.shots),
        "norm_opt_demand_gain_vs_raw_opt": float(norm_cap.total_demand - raw_cap.demand),
        "ratio_optimization_norm_gain": float(
            raw_cap.ammo_norm - norm_cap.ammo_norm
        ),
        "selected_demand_minus_raw_opt_demand": float(
            selected_demand - raw_cap.demand
        ),
        "selected_demand_minus_norm_opt_demand": float(
            selected_demand - norm_cap.total_demand
        ),
    }

    target_rows: List[Dict[str, object]] = []
    for target, selected_scheme in enumerate(scheme_idx.tolist()):
        selected = table[target][int(selected_scheme)]
        raw_scheme = int(raw_cap.scheme_idx[target])
        norm_scheme = int(norm_cap.scheme_idx[target])
        raw_item = table[target][raw_scheme]
        norm_item = table[target][norm_scheme]
        actual_target_shots = int(np.sum(A[:, :, target]))

        selected_cap = int(selected.capacity_shots)
        raw_cap_target = int(raw_item.capacity_shots)
        target_rows.append(
            {
                "target": int(target),
                "selected_scheme": int(selected_scheme),
                "raw_opt_scheme": raw_scheme,
                "norm_opt_scheme": norm_scheme,
                "actual_target_raw_ammo": actual_target_shots,
                "selected_demand": float(selected.demand_total),
                "raw_opt_demand": float(raw_item.demand_total),
                "norm_opt_demand": float(norm_item.demand_total),
                "selected_capacity_lb_shots": selected_cap,
                "raw_opt_capacity_lb_shots": raw_cap_target,
                "norm_opt_capacity_lb_shots": int(norm_item.capacity_shots),
                "target_allocation_gap_shots": int(
                    max(actual_target_shots - selected_cap, 0)
                ),
                "target_raw_scheme_gap_shots": int(
                    max(selected_cap - raw_cap_target, 0)
                ),
                "target_raw_scheme_gap_ratio": float(
                    selected_cap / max(float(raw_cap_target), 1e-12) - 1.0
                ),
                "selected_matches_raw_opt": bool(selected_scheme == raw_scheme),
                "selected_matches_norm_opt": bool(selected_scheme == norm_scheme),
                "raw_opt_matches_norm_opt": bool(raw_scheme == norm_scheme),
                "demand_change_if_raw_opt": float(
                    raw_item.demand_total - selected.demand_total
                ),
                "demand_change_if_norm_opt": float(
                    norm_item.demand_total - selected.demand_total
                ),
                "selected_capacity_implied_norm": float(
                    selected_cap / max(float(selected.demand_total), 1e-12)
                ),
                "raw_opt_capacity_implied_norm": float(
                    raw_cap_target / max(float(raw_item.demand_total), 1e-12)
                ),
                "norm_opt_capacity_implied_norm": float(
                    int(norm_item.capacity_shots)
                    / max(float(norm_item.demand_total), 1e-12)
                ),
            }
        )

    return scenario, target_rows
