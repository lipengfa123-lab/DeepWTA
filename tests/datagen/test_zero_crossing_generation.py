from __future__ import annotations

import numpy as np

from deepwta.data.zero_crossing import (
    balanced_non_crossing_matching,
    build_reference_solution_zero_crossing,
    build_zero_crossing_target_base_map,
    reference_with_inventory_budget_zero_crossing,
)
from deepwta.utils.geometry import count_crossings, find_crossing_pairs


def test_balanced_matching_is_exactly_non_crossing() -> None:
    # Two separated deployment regions with deliberately irregular coordinates.
    base = np.array(
        [[40, 50], [180, 210], [330, 100], [510, 260], [700, 80], [820, 300]],
        dtype=np.float64,
    )
    target = np.array(
        [[60, 1200], [230, 1510], [290, 1120], [550, 1430], [690, 1660], [840, 1300]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(2026)
    matching = balanced_non_crossing_matching(base, target, rng)
    assert matching is not None
    assert len(matching) == len(base)
    assert find_crossing_pairs(base, target, matching) == []


def test_target_base_map_has_zero_exact_crossings() -> None:
    rng = np.random.default_rng(7)
    base = np.column_stack([
        rng.integers(30, 880, size=20),
        rng.integers(30, 900, size=20),
    ]).astype(np.float64)
    target = np.column_stack([
        rng.integers(30, 880, size=20),
        rng.integers(1050, 1680, size=20),
    ]).astype(np.float64)
    target_to_base = build_zero_crossing_target_base_map(base, target, rng)
    assert target_to_base is not None
    routes = [(int(target_to_base[j]), j) for j in range(len(target))]
    assert find_crossing_pairs(base, target, routes) == []


def _toy_problem(num_targets: int = 8):
    num_weapons = 3
    num_bases = num_targets
    num_schemes = 3
    base = np.column_stack([
        np.linspace(50, 800, num_bases),
        np.linspace(100, 350, num_bases),
    ]).astype(np.float64)
    target = np.column_stack([
        np.linspace(70, 820, num_targets),
        np.linspace(1250, 1550, num_targets),
    ]).astype(np.float64)
    rng = np.random.default_rng(9)
    target_to_base = build_zero_crossing_target_base_map(base, target, rng)
    assert target_to_base is not None

    q = np.full((num_weapons, num_bases, num_targets), 0.10, dtype=np.float32)
    x = np.ones_like(q, dtype=np.float32)
    u = np.zeros((num_schemes, num_weapons, num_targets), dtype=np.int16)
    g = np.zeros((num_schemes, num_targets), dtype=np.int8)
    resource = np.zeros((num_schemes, num_targets), dtype=np.float32)
    for target_id in range(num_targets):
        # Light/Medium/Heavy with strictly increasing expected resource.
        u[0, 0, target_id] = 1
        u[1, 0:2, target_id] = 1
        u[2, 0:3, target_id] = 1
        g[:, target_id] = [1, 2, 3]
        resource[:, target_id] = [1.2, 2.4, 3.6]
    return base, target, q, x, u, g, resource, target_to_base


def test_zero_crossing_reference_is_hard_feasible() -> None:
    base, target, q, x, u, g, resource, target_to_base = _toy_problem()
    rng = np.random.default_rng(10)
    thresholds = np.ones(len(target), dtype=np.int8)
    result = build_reference_solution_zero_crossing(
        u,
        g,
        thresholds,
        resource,
        q,
        x,
        inventory_max=80,
        reference_capacity_ratio=0.72,
        target_to_base=target_to_base,
        rng=rng,
    )
    assert result is not None
    selected, allocation = result
    assert count_crossings(allocation, base, target) == 0
    chosen = u[selected, :, np.arange(len(target))].T
    effective = np.sum(allocation * (1.0 - q), axis=1)
    assert np.all(effective + 1e-6 >= chosen)
    assert np.all((allocation > 0) <= (x > 0.5))


def test_fixed_budget_logic_retains_zero_crossing() -> None:
    base, target, q, x, u, g, resource, target_to_base = _toy_problem()
    rng = np.random.default_rng(11)
    sampled_thresholds = np.array([1, 2, 1, 3, 1, 2, 1, 1], dtype=np.int8)
    result = reference_with_inventory_budget_zero_crossing(
        u,
        g,
        sampled_thresholds,
        resource,
        q,
        x,
        inventory_max=80,
        reference_capacity_ratio=0.72,
        total_weapons=40,
        target_to_base=target_to_base,
        rng=rng,
    )
    assert result is not None
    thresholds, selected, allocation = result
    assert int(np.sum(allocation)) <= 40
    assert np.all(g[selected, np.arange(len(target))] >= thresholds)
    assert count_crossings(allocation, base, target) == 0
