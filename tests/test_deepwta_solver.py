from __future__ import annotations

import numpy as np

from configs.deepwta_solver_config import DeepWTASolverConfig
from deepwta._feasible_construction import construct_feasible_solution
from deepwta.mechanisms.conflict_index import LazyConflictIndex
from deepwta.mechanisms.integer_decoder import (
    build_target_route_candidates,
    floor_integerize,
    residual_aware_integerize,
)
from deepwta.solver import solve_deepwta
from deepwta.utils.geometry import find_crossing_pairs
from deepwta.utils.validator import validate_solution


def build_case():
    K, N, M, L = 3, 4, 3, 2
    B = np.full((K, N), 8.0)
    Q = np.array(
        [
            [[0.10, 0.15, 0.12], [0.20, 0.10, 0.14], [0.05, 0.25, 0.09], [0.13, 0.11, 0.16]],
            [[0.12, 0.08, 0.10], [0.18, 0.12, 0.15], [0.10, 0.20, 0.11], [0.09, 0.16, 0.13]],
            [[0.11, 0.13, 0.09], [0.17, 0.10, 0.12], [0.08, 0.18, 0.10], [0.10, 0.14, 0.15]],
        ],
        dtype=float,
    )
    X = np.ones((K, N, M), dtype=float)

    U = np.zeros((L, K, M), dtype=float)
    U[0, :, :] = np.array(
        [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    )
    U[1, :, :] = np.array(
        [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    )

    G = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    G_th = np.array([1.0, 1.0, 1.0])
    scheme_probs = np.array([[0.9, 0.1], [0.85, 0.15], [0.88, 0.12]])

    # Diffuse fractional allocation, but each target has a clear aggregate
    # route preference shared across weapons.
    A_soft = np.array(
        [
            [[0.82, 0.10, 0.12], [0.70, 0.78, 0.14], [0.18, 0.20, 0.72], [0.12, 0.15, 0.20]],
            [[0.55, 0.08, 0.10], [0.48, 0.56, 0.12], [0.10, 0.14, 0.52], [0.08, 0.12, 0.14]],
            [[0.45, 0.07, 0.08], [0.40, 0.46, 0.10], [0.09, 0.12, 0.43], [0.07, 0.10, 0.12]],
        ],
        dtype=float,
    )

    base_coords = np.array([[0, 0], [0, 10], [0, 20], [0, 30]], dtype=float)
    target_coords = np.array([[10, 30], [10, 15], [10, 0]], dtype=float)
    return A_soft, scheme_probs, B, U, G, G_th, Q, X, base_coords, target_coords


def test_floor_is_inventory_safe():
    A_soft, _, B, _, _, _, _, X, _, _ = build_case()
    floor_A, residual, report = floor_integerize(A_soft, X, B)
    assert np.all(floor_A.sum(axis=2) <= B + 1e-9)
    assert np.all(residual >= -1e-12)
    assert np.all(residual < 1.0 + 1e-12)
    assert report["floor_inventory_clipped_shots"] == 0


def test_target_candidates_are_route_level():
    A_soft, _, _, _, _, _, _, X, _, _ = build_case()
    candidates = build_target_route_candidates(
        A_soft,
        X,
        use_topk=True,
        rho=0.90,
        k_min=2,
        k_max=3,
        tau=0.25,
    )
    assert set(candidates) == {0, 1, 2}
    assert all(1 <= c.active_count <= 3 for c in candidates.values())


def test_residual_aware_seed_is_deterministic_and_physically_safe():
    A_soft, _, B, U, _, _, Q, X, _, _ = build_case()
    selected_demand = U[0]
    floor_seed, floor_reference, _, floor_report = residual_aware_integerize(
        A_soft, X, B, selected_demand, Q, mode="floor"
    )
    expected_floor, _, _ = floor_integerize(A_soft, X, B)
    np.testing.assert_array_equal(floor_seed, expected_floor)
    np.testing.assert_array_equal(floor_reference, expected_floor)
    assert floor_report["integer_seed_roundup_shots"] == 0

    for mode in ("ra_r", "ra_s", "ra_n"):
        seed_a, floor_a, _, report_a = residual_aware_integerize(
            A_soft, X, B, selected_demand, Q, mode=mode
        )
        seed_b, _, _, report_b = residual_aware_integerize(
            A_soft, X, B, selected_demand, Q, mode=mode
        )
        np.testing.assert_array_equal(seed_a, seed_b)
        assert np.all(seed_a >= floor_a)
        assert np.all(seed_a - floor_a <= 1)
        assert np.all(seed_a.sum(axis=2) <= np.floor(B + 1e-7))
        assert np.sum(seed_a * (X <= 0.5)) == 0
        assert report_a["integer_seed_inventory_violation"] == 0
        assert report_a["integer_seed_reachability_violation"] == 0
        assert report_a["integer_seed_shortage"] <= floor_report["integer_seed_shortage"]
        for key in set(report_a) - {"integer_seed_rounding_kernel_s"}:
            assert report_a[key] == report_b[key]


def test_residual_rounding_never_adds_to_floor_satisfied_pair():
    soft = np.array([[[1.9, 0.9]]], dtype=float)
    X = np.ones_like(soft)
    B = np.array([[3.0]])
    demand = np.array([[0.5, 1.0]])
    Q = np.zeros_like(soft)
    seed, floor_seed, _, _ = residual_aware_integerize(
        soft, X, B, demand, Q, mode="ra_r"
    )
    # Target 0 is already satisfied by floor(1.9)=1; only target 1 may round.
    assert seed[0, 0, 0] == floor_seed[0, 0, 0] == 1
    assert seed[0, 0, 1] == floor_seed[0, 0, 1] + 1


def test_global_budget_prevents_duplicate_shortage_compensation():
    soft = np.array([[[0.9], [0.8], [0.7]]], dtype=float)
    X = np.ones_like(soft)
    B = np.ones((1, 3), dtype=float)
    demand = np.array([[0.8]])
    Q = np.full_like(soft, 0.2)
    local, _, _, local_report = residual_aware_integerize(
        soft, X, B, demand, Q, mode="ra_s"
    )
    global_seed, floor_seed, _, global_report = residual_aware_integerize(
        soft, X, B, demand, Q, mode="gb_s"
    )
    assert np.sum(local - floor_seed) == 3
    assert np.sum(global_seed - floor_seed) == 1
    assert global_report["integer_seed_oversupplied_effective_demand"] == 0.0
    assert global_report["integer_seed_rounding_efficiency"] > local_report[
        "integer_seed_rounding_efficiency"
    ]


def test_global_budget_prunes_base_inventory_conflict():
    soft = np.array([[[0.9, 0.8]]], dtype=float)
    X = np.ones_like(soft)
    B = np.array([[1.0]])
    demand = np.array([[1.0, 1.0]])
    Q = np.zeros_like(soft)
    seed, floor_seed, _, report = residual_aware_integerize(
        soft, X, B, demand, Q, mode="gb_s"
    )
    assert np.sum(seed - floor_seed) == 1
    assert np.sum(seed, axis=2)[0, 0] == 1
    assert report["integer_seed_base_pruned_roundups"] == 1
    assert report["integer_seed_inventory_violation"] == 0


def test_global_route_reuse_prefers_floor_active_route():
    soft = np.array([[[1.1], [0.9]]], dtype=float)
    X = np.ones_like(soft)
    B = np.array([[2.0, 1.0]])
    demand = np.array([[1.5]])
    Q = np.zeros_like(soft)
    gb_s, floor_seed, _, _ = residual_aware_integerize(
        soft, X, B, demand, Q, mode="gb_s"
    )
    gb_sr, _, _, report = residual_aware_integerize(
        soft, X, B, demand, Q, mode="gb_sr"
    )
    assert floor_seed[0, 0, 0] == 1
    assert gb_s[0, 1, 0] == 1
    assert gb_sr[0, 0, 0] == 2
    assert report["integer_seed_new_routes"] == 0


def test_solver_seed_mode_is_opt_in_and_reported():
    args = build_case()
    baseline, _, baseline_report = solve_deepwta(
        *args,
        config=DeepWTASolverConfig(fallback_to_construction=False, max_passes=0),
        return_report=True,
    )
    ra, _, ra_report = solve_deepwta(
        *args,
        config=DeepWTASolverConfig(
            fallback_to_construction=False,
            max_passes=0,
            integer_seed_mode="ra_s",
        ),
        return_report=True,
    )
    assert baseline_report["integer_seed_mode"] == "floor"
    assert ra_report["integer_seed_mode"] == "ra_s"
    assert baseline_report["integer_seed_roundup_shots"] == 0
    assert ra_report["integer_seed_roundup_shots"] > 0
    assert baseline_report["hard_feasible"] and ra_report["hard_feasible"]
    assert np.all(baseline >= 0) and np.all(ra >= 0)


def test_seed_preserving_constructor_trims_roundup_oversupply():
    A_soft = np.array([[[0.9], [0.8]]], dtype=float)
    scheme_probs = np.array([[1.0]])
    B = np.array([[1.0, 1.0]])
    U = np.array([[[1.0]]])
    G = np.array([[1.0]])
    G_th = np.array([1.0])
    Q = np.zeros_like(A_soft)
    X = np.ones_like(A_soft)
    base_coords = np.array([[0.0, 0.0], [0.0, 1.0]])
    target_coords = np.array([[1.0, 0.0]])
    seed = np.ones_like(A_soft, dtype=np.int64)
    floor = np.zeros_like(seed)
    allocation, schemes, report = construct_feasible_solution(
        A_soft, scheme_probs, B, U, G, G_th, Q, X,
        base_coords, target_coords,
        return_report=True,
        initial_allocation=seed,
        initial_floor_allocation=floor,
        initial_scheme_idx=np.array([0]),
    )
    exact = validate_solution(
        allocation, schemes, B, U, G, G_th, Q, X,
        base_coords, target_coords,
    )
    assert exact["hard_feasible"], exact
    assert report["seed_preserving_fallback_requested"]
    assert report["seed_preserving_fallback_succeeded"]
    assert report["seed_preserving_trimmed_shots"] == 1
    assert np.sum(allocation) == 1


def test_lazy_conflict_index_matches_exact():
    *_, base_coords, target_coords = build_case()
    routes = [(0, 0), (1, 1), (2, 2), (3, 0)]
    index = LazyConflictIndex(base_coords, target_coords, routes[:2])
    active = set(routes[:2])
    for route in routes[2:]:
        index.query_delta_open(route, active)
        index.register_route(route)
        active.add(route)

    exact = find_crossing_pairs(base_coords, target_coords, sorted(active))
    assert index.crossing_count(active) == len(exact)


def test_standard_mode_is_exact_feasible_without_fallback():
    args = build_case()
    A_soft, scheme_probs, B, U, G, G_th, Q, X, base_coords, target_coords = args
    A, schemes, report = solve_deepwta(
        *args,
        config=DeepWTASolverConfig(fallback_to_construction=False, max_passes=3),
        return_report=True,
    )
    exact = validate_solution(
        A, schemes, B, U, G, G_th, Q, X, base_coords, target_coords,
    )
    assert exact["hard_feasible"], exact
    assert not report["construction_fallback_used"]
    assert int(report["final_components"]["crossing_count"]) == int(
        exact["crossing_conflict_count"]
    )


if __name__ == "__main__":
    test_floor_is_inventory_safe()
    test_target_candidates_are_route_level()
    test_lazy_conflict_index_matches_exact()
    test_standard_mode_is_exact_feasible_without_fallback()
    print("PASS: DeepWTA solver synthetic tests")
