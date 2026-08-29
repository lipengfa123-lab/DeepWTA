from __future__ import annotations

import itertools
import numpy as np

from deepwta.evaluation.ammo_lower_bound import (
    audit_solution_ammo,
    build_target_scheme_bounds,
    minimize_fractional_bound,
)


def test_capacity_bound_uses_high_survival_first() -> None:
    B = np.array([[1, 10]], dtype=float)
    U = np.array([[[1.5]]], dtype=float)  # [L,K,M]
    G = np.array([[1.0]])
    G_th = np.array([0.5])
    Q = np.array([[[0.1], [0.5]]], dtype=float)  # survivals .9, .5
    X = np.ones_like(Q)
    table = build_target_scheme_bounds(B, U, G, G_th, Q, X)
    item = table[0][0]
    assert item.unlimited_shots == 2
    assert item.capacity_shots == 3
    assert item.local_inventory_penalty_shots == 1


def test_fractional_bound_matches_bruteforce() -> None:
    B = np.full((1, 1), 100.0)
    # target0 schemes: (shots,demand) -> about (2,1.0), (3,2.0)
    # target1 schemes: about (2,1.2), (4,3.0)
    U = np.array([
        [[1.0, 1.2]],
        [[2.0, 3.0]],
    ], dtype=float)
    G = np.ones((2, 2), dtype=float)
    G_th = np.zeros(2, dtype=float)
    Q = np.array([[[0.4, 0.4]]], dtype=float)  # survival .6
    X = np.ones_like(Q)
    table = build_target_scheme_bounds(B, U, G, G_th, Q, X)
    result = minimize_fractional_bound(table, "capacity_shots")

    brute = None
    for schemes in itertools.product(range(2), repeat=2):
        shots = sum(table[m][l].capacity_shots for m, l in enumerate(schemes))
        demand = sum(table[m][l].demand_total for m, l in enumerate(schemes))
        ratio = shots / demand
        key = (ratio, shots, tuple(schemes))
        if brute is None or key < brute:
            brute = key
    assert abs(result.ammo_norm - brute[0]) < 1e-12


def test_audit_decomposition_and_near_lb() -> None:
    B = np.array([[10]], dtype=float)
    U = np.array([[[1.0]]], dtype=float)
    G = np.array([[1.0]])
    G_th = np.array([0.5])
    Q = np.array([[[0.0]]], dtype=float)
    X = np.ones_like(Q)
    table = build_target_scheme_bounds(B, U, G, G_th, Q, X)
    A = np.array([[[1]]], dtype=np.int64)
    audit, targets = audit_solution_ammo(A, np.array([0]), table)
    assert audit["actual_ammo"] == 1
    assert abs(audit["actual_ammo_norm"] - 1.0) < 1e-12
    assert abs(audit["global_capacity_lb_norm"] - 1.0) < 1e-12
    assert audit["near_capacity_lb"] is True
    assert audit["bottleneck_class"] == "near_relaxed_lb"
    assert targets[0]["target_allocation_gap_shots"] == 0


def test_invalid_scheme_is_excluded() -> None:
    B = np.array([[10]], dtype=float)
    U = np.array([[[1.0]], [[0.1]]], dtype=float)
    G = np.array([[1.0], [0.1]])
    G_th = np.array([0.5])
    Q = np.array([[[0.0]]], dtype=float)
    X = np.ones_like(Q)
    table = build_target_scheme_bounds(B, U, G, G_th, Q, X)
    result = minimize_fractional_bound(table)
    assert result.scheme_idx.tolist() == [0]


def main() -> None:
    test_capacity_bound_uses_high_survival_first()
    test_fractional_bound_matches_bruteforce()
    test_audit_decomposition_and_near_lb()
    test_invalid_scheme_is_excluded()
    print("PASS: Ammo Lower-Bound & Bottleneck Audit tests")


if __name__ == "__main__":
    main()
