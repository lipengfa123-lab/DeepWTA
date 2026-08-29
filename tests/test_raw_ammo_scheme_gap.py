from __future__ import annotations

import numpy as np

from deepwta.evaluation.ammo_lower_bound import build_target_scheme_bounds
from deepwta.evaluation.raw_ammo_scheme_gap import (
    audit_raw_ammo_scheme_gap,
    minimum_raw_choice,
)


def test_raw_gap_exact_decomposition() -> None:
    # One target, two legal schemes, survival=0.6.
    # scheme0 demand 0.5 -> 1 shot (raw optimum)
    # scheme1 demand 1.1 -> 2 shots (selected)
    B = np.array([[10.0]])
    U = np.array([[[0.5]], [[1.1]]], dtype=float)
    G = np.ones((2, 1), dtype=float)
    G_th = np.zeros(1, dtype=float)
    Q = np.array([[[0.4]]], dtype=float)
    X = np.ones_like(Q)
    table = build_target_scheme_bounds(B, U, G, G_th, Q, X)

    A = np.array([[[3]]], dtype=np.int64)
    audit, targets = audit_raw_ammo_scheme_gap(A, np.array([1]), table)

    assert audit["actual_raw_ammo"] == 3
    assert audit["selected_scheme_capacity_lb_raw"] == 2
    assert audit["global_raw_capacity_lb_shots"] == 1
    assert audit["global_raw_unlimited_lb_shots"] == 1
    assert audit["raw_allocation_cross_target_gap_shots"] == 1
    assert audit["raw_scheme_choice_gap_shots"] == 1
    assert audit["raw_local_inventory_gap_shots"] == 0
    assert audit["raw_total_gap_to_unlimited_lb_shots"] == 2
    assert abs(audit["raw_scheme_gap_share"] - 0.5) < 1e-12
    assert targets[0]["raw_opt_scheme"] == 0


def test_denominator_effect_is_visible() -> None:
    # Raw-shot optimum chooses 1-shot low-demand scheme, while ammo_norm optimum
    # chooses 2-shot high-demand scheme because 2/1.1 < 1/0.5.
    B = np.array([[10.0]])
    U = np.array([[[0.5]], [[1.1]]], dtype=float)
    G = np.ones((2, 1), dtype=float)
    G_th = np.zeros(1, dtype=float)
    Q = np.array([[[0.4]]], dtype=float)
    X = np.ones_like(Q)
    table = build_target_scheme_bounds(B, U, G, G_th, Q, X)

    raw = minimum_raw_choice(table, "capacity_shots")
    A = np.array([[[2]]], dtype=np.int64)
    audit, _ = audit_raw_ammo_scheme_gap(A, np.array([1]), table)

    assert raw.scheme_idx.tolist() == [0]
    assert audit["raw_opt_vs_norm_opt_target_match_rate"] == 0.0
    assert audit["norm_opt_shot_premium_vs_raw_opt"] == 1
    assert audit["norm_opt_demand_gain_vs_raw_opt"] > 0.0
    assert audit["ratio_optimization_norm_gain"] > 0.0


def test_local_inventory_component() -> None:
    # Best route has survival .9 but only one shot; second route has .5.
    # Demand 1.5 => unlimited LB 2, capacity-aware LB 3.
    B = np.array([[1.0, 10.0]])
    U = np.array([[[1.5]]], dtype=float)
    G = np.ones((1, 1), dtype=float)
    G_th = np.zeros(1, dtype=float)
    Q = np.array([[[0.1], [0.5]]], dtype=float)
    X = np.ones_like(Q)
    table = build_target_scheme_bounds(B, U, G, G_th, Q, X)
    A = np.array([[[1], [2]]], dtype=np.int64)
    audit, _ = audit_raw_ammo_scheme_gap(A, np.array([0]), table)

    assert audit["global_raw_unlimited_lb_shots"] == 2
    assert audit["global_raw_capacity_lb_shots"] == 3
    assert audit["raw_local_inventory_gap_shots"] == 1


def test_near_raw_lb_classification() -> None:
    B = np.array([[10.0]])
    U = np.array([[[1.0]]], dtype=float)
    G = np.ones((1, 1), dtype=float)
    G_th = np.zeros(1, dtype=float)
    Q = np.zeros((1, 1, 1), dtype=float)
    X = np.ones_like(Q)
    table = build_target_scheme_bounds(B, U, G, G_th, Q, X)
    A = np.array([[[1]]], dtype=np.int64)
    audit, _ = audit_raw_ammo_scheme_gap(A, np.array([0]), table)
    assert audit["near_raw_capacity_lb"] is True
    assert audit["raw_bottleneck_class"] == "near_raw_relaxed_lb"


def main() -> None:
    test_raw_gap_exact_decomposition()
    test_denominator_effect_is_visible()
    test_local_inventory_component()
    test_near_raw_lb_classification()
    print("PASS: Raw-Ammo Scheme Gap Audit tests")


if __name__ == "__main__":
    main()
