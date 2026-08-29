from __future__ import annotations

from typing import Any

import numpy as np

from deepwta.utils.validator import selected_plan, validate_solution


def gini_coefficient(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = np.maximum(x, 0.0)
    total = float(np.sum(x))
    if x.size == 0 or total <= 1e-12:
        return 0.0

    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1, dtype=float)
    return float(
        (2.0 * np.sum(index * x) / (n * total))
        - (n + 1.0) / n
    )


def size_bucket(num_bases: int, num_targets: int) -> str:
    size = max(int(num_bases), int(num_targets))
    if size <= 20:
        return "small"
    if size <= 40:
        return "medium"
    if size <= 60:
        return "large"
    return "xlarge"


def scenario_metrics(
    allocation: np.ndarray,
    scheme_idx: np.ndarray,
    B: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    G_th: np.ndarray,
    Q: np.ndarray,
    X: np.ndarray,
    D: np.ndarray,
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    tol: float = 1e-7,
) -> dict[str, Any]:
    A = np.asarray(allocation, dtype=float)
    B = np.asarray(B, dtype=float)
    U = np.asarray(U, dtype=float)
    G = np.asarray(G, dtype=float)
    G_th = np.asarray(G_th, dtype=float)
    Q = np.asarray(Q, dtype=float)
    X = np.asarray(X, dtype=float)
    D = np.asarray(D, dtype=float)
    base_coords = np.asarray(base_coords, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)
    scheme_idx = np.asarray(scheme_idx, dtype=np.int64)

    K, N, M = Q.shape
    report = validate_solution(
        A,
        scheme_idx,
        B,
        U,
        G,
        G_th,
        Q,
        X,
        base_coords,
        target_coords,
        tol=tol,
    )

    selected_demand, selected_damage = selected_plan(
        U,
        G,
        scheme_idx,
        tol=tol,
    )
    demand_total = max(float(np.sum(np.maximum(selected_demand, 0.0))), 1e-12)

    positive = np.maximum(A, 0.0)
    ammo = float(np.sum(positive))
    safe_ammo = max(ammo, 1e-12)

    max_d = max(float(np.max(D)), 1e-12)
    d_norm = D / max_d

    avg_distance = float(
        np.sum(positive * d_norm[None, :, :]) / safe_ammo
    ) if ammo > tol else 0.0
    avg_interception = float(
        np.sum(positive * Q) / safe_ammo
    ) if ammo > tol else 0.0

    base_load = positive.sum(axis=(0, 2))
    active_base = base_load > tol
    active_base_count = int(np.sum(active_base))
    active_base_ratio = float(active_base_count / max(N, 1))
    mean_active_base_load = (
        float(np.mean(base_load[active_base]))
        if active_base_count > 0
        else 0.0
    )
    max_base_load_ratio = (
        float(np.max(base_load) / safe_ammo)
        if ammo > tol
        else 0.0
    )

    route_load = positive.sum(axis=0)
    active_route_mask = route_load > tol
    active_route_count = int(np.sum(active_route_mask))
    active_route_ratio = float(active_route_count / max(N * M, 1))

    # Normalize exact crossings by all unordered active-route pairs that are
    # eligible to conflict under the project definition (distinct base and target).
    active_rc = np.argwhere(active_route_mask)
    total_route_pairs = active_route_count * (active_route_count - 1) // 2
    if active_route_count > 0:
        per_base = np.bincount(active_rc[:, 0], minlength=N)
        per_target = np.bincount(active_rc[:, 1], minlength=M)
        shared_base_pairs = int(np.sum(per_base * (per_base - 1) // 2))
        shared_target_pairs = int(np.sum(per_target * (per_target - 1) // 2))
    else:
        shared_base_pairs = 0
        shared_target_pairs = 0

    eligible_active_route_pairs = max(
        int(total_route_pairs - shared_base_pairs - shared_target_pairs),
        0,
    )
    exact_crossing_count = float(
        report.get("crossing_conflict_count", np.nan)
    )
    crossing_pair_rate = (
        float(exact_crossing_count / eligible_active_route_pairs)
        if eligible_active_route_pairs > 0
        else 0.0
    )
    crossings_per_active_route = (
        float(exact_crossing_count / active_route_count)
        if active_route_count > 0
        else 0.0
    )

    inventory_use = positive.sum(axis=2)
    inventory_cap = np.maximum(B, 0.0)
    positive_inventory = inventory_cap > tol
    mean_inventory_utilization = (
        float(np.mean(
            inventory_use[positive_inventory]
            / np.maximum(inventory_cap[positive_inventory], 1e-12)
        ))
        if np.any(positive_inventory)
        else 0.0
    )
    max_inventory_utilization = (
        float(np.max(
            inventory_use[positive_inventory]
            / np.maximum(inventory_cap[positive_inventory], 1e-12)
        ))
        if np.any(positive_inventory)
        else 0.0
    )

    chosen_levels = selected_damage.astype(int)
    level1 = float(np.mean(chosen_levels == 1))
    level2 = float(np.mean(chosen_levels == 2))
    level3 = float(np.mean(chosen_levels == 3))

    return {
        "hard_feasible": float(bool(report.get("hard_feasible", False))),
        "inventory_feasible": float(bool(report.get("inventory_feasible", False))),
        "demand_feasible": float(bool(report.get("demand_feasible", False))),
        "level_feasible": float(bool(report.get("level_feasible", False))),
        "reachability_feasible": float(bool(report.get("reachability_feasible", False))),
        "integer_feasible": float(bool(report.get("is_integer", False))),
        "ammo": ammo,
        "ammo_norm": float(ammo / demand_total),
        "crossing_count": exact_crossing_count,
        "eligible_active_route_pairs": float(eligible_active_route_pairs),
        "crossing_pair_rate": crossing_pair_rate,
        "crossings_per_active_route": crossings_per_active_route,
        "active_route_count": float(active_route_count),
        "active_route_ratio": active_route_ratio,
        "avg_distance": avg_distance,
        "avg_interception": avg_interception,
        "demand_violation": float(report.get("demand_violation", np.nan)),
        "max_demand_shortfall": float(report.get("max_demand_shortfall", np.nan)),
        "inventory_violation": float(report.get("inventory_violation", np.nan)),
        "max_inventory_violation": float(report.get("max_inventory_violation", np.nan)),
        "reachability_violation": float(report.get("reachability_violation", np.nan)),
        "active_base_count": float(active_base_count),
        "active_base_ratio": active_base_ratio,
        "mean_active_base_load": mean_active_base_load,
        "max_base_load_ratio": max_base_load_ratio,
        "base_load_gini": gini_coefficient(base_load),
        "mean_inventory_utilization": mean_inventory_utilization,
        "max_inventory_utilization": max_inventory_utilization,
        "mean_damage_level": float(np.mean(selected_damage)),
        "level1_ratio": level1,
        "level2_ratio": level2,
        "level3_ratio": level3,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int = 2026,
    samples: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        value = float(x[0])
        return value, value

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, x.size, size=(samples, x.size))
    means = x[draws].mean(axis=1)
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lower, upper
