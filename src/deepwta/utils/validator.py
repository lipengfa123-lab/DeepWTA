from __future__ import annotations

from typing import Any

import numpy as np

from deepwta.utils.geometry import active_routes, crossing_pairs_from_allocations


def _valid_integer_indices(values: np.ndarray, upper: int, tol: float) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("scheme_idx must be one-dimensional")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError("scheme_idx must be numeric")

    as_float = raw.astype(float)
    if not np.all(np.isfinite(as_float)):
        raise ValueError("scheme_idx contains non-finite values")

    rounded = np.rint(as_float)
    if np.any(np.abs(as_float - rounded) > tol):
        raise ValueError("scheme_idx must contain integer indices")

    indices = rounded.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= upper):
        raise ValueError("scheme_idx contains an invalid scheme index")
    return indices


def selected_plan(
    U: np.ndarray,
    G: np.ndarray,
    scheme_idx: np.ndarray,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(U, dtype=float)
    G = np.asarray(G, dtype=float)

    if U.ndim != 3:
        raise ValueError("U must have shape [L, K, M]")
    L, _, M = U.shape
    if G.shape != (L, M):
        raise ValueError(f"G must have shape [{L}, {M}]")

    indices = _valid_integer_indices(scheme_idx, L, tol)
    if indices.shape != (M,):
        raise ValueError(f"scheme_idx must have shape [{M}]")

    S = U[indices, :, np.arange(M)].T
    G_chosen = G[indices, np.arange(M)]
    return S, G_chosen


def validate_fixed_plan(
    allocations: np.ndarray,
    B: np.ndarray,
    S_chosen: np.ndarray,
    G_chosen: np.ndarray,
    G_th: np.ndarray,
    Q: np.ndarray,
    X: np.ndarray,
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    tol: float = 1e-7,
) -> dict[str, Any]:
    A = np.asarray(allocations, dtype=float)
    B = np.asarray(B, dtype=float)
    S = np.asarray(S_chosen, dtype=float)
    G_chosen = np.asarray(G_chosen, dtype=float)
    G_th = np.asarray(G_th, dtype=float)
    Q = np.asarray(Q, dtype=float)
    X = np.asarray(X, dtype=float)
    base_coords = np.asarray(base_coords, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)

    if Q.ndim != 3:
        raise ValueError("Q must have shape [K, N, M]")

    K, N, M = Q.shape
    expected_shapes = {
        "allocations": (K, N, M),
        "B": (K, N),
        "S_chosen": (K, M),
        "G_chosen": (M,),
        "G_th": (M,),
        "X": (K, N, M),
    }
    actual_shapes = {
        "allocations": A.shape,
        "B": B.shape,
        "S_chosen": S.shape,
        "G_chosen": G_chosen.shape,
        "G_th": G_th.shape,
        "X": X.shape,
    }

    coords_valid = (
        base_coords.ndim == 2
        and target_coords.ndim == 2
        and base_coords.shape[0] == N
        and target_coords.shape[0] == M
        and base_coords.shape[1] >= 2
        and target_coords.shape[1] >= 2
    )
    shape_valid = expected_shapes == actual_shapes and coords_valid
    if not shape_valid:
        return {
            "is_feasible": False,
            "hard_feasible": False,
            "shape_valid": False,
            "expected_shapes": expected_shapes,
            "actual_shapes": actual_shapes,
            "coords_valid": bool(coords_valid),
        }

    finite_valid = all(
        np.all(np.isfinite(array))
        for array in (A, B, S, G_chosen, G_th, Q, X, base_coords, target_coords)
    )
    q_domain_valid = bool(np.all((Q >= -tol) & (Q <= 1.0 + tol)))
    x_domain_valid = bool(np.all((X >= -tol) & (X <= 1.0 + tol)))

    rounded = np.rint(A)
    negative_gap = np.maximum(-A, 0.0)
    negative_violation = float(negative_gap.sum())
    max_negative_violation = float(negative_gap.max(initial=0.0))
    integer_violation = float(np.abs(A - rounded).max(initial=0.0))

    ammo_by_base = A.sum(axis=2)
    inventory_gap = np.maximum(ammo_by_base - B, 0.0)

    effective_hits = np.sum(A * (1.0 - Q), axis=1)
    demand_gap = np.maximum(S - effective_hits, 0.0)

    level_gap = np.maximum(G_th - G_chosen, 0.0)
    unreachable_mask = X <= 0.5
    unreachable_shots = float(np.sum(np.maximum(A, 0.0) * unreachable_mask))

    crossing_pairs = crossing_pairs_from_allocations(
        A,
        base_coords,
        target_coords,
        tol=tol,
    )

    inventory_violation = float(inventory_gap.sum())
    demand_violation = float(demand_gap.sum())
    level_violation = float(level_gap.sum())
    objective = float(A.sum()) if np.all(np.isfinite(A)) else float("nan")

    is_nonnegative = max_negative_violation <= tol
    is_integer = integer_violation <= tol
    inventory_feasible = float(inventory_gap.max(initial=0.0)) <= tol
    demand_feasible = float(demand_gap.max(initial=0.0)) <= tol
    level_feasible = float(level_gap.max(initial=0.0)) <= tol
    reachability_feasible = unreachable_shots <= tol

    # Geometric crossing is a soft objective, not a hard feasibility condition.
    hard_feasible = all(
        [
            finite_valid,
            q_domain_valid,
            x_domain_valid,
            is_nonnegative,
            is_integer,
            inventory_feasible,
            demand_feasible,
            level_feasible,
            reachability_feasible,
        ]
    )

    return {
        "is_feasible": bool(hard_feasible),
        "hard_feasible": bool(hard_feasible),
        "shape_valid": True,
        "coords_valid": True,
        "finite_valid": bool(finite_valid),
        "q_domain_valid": q_domain_valid,
        "x_domain_valid": x_domain_valid,
        "is_nonnegative": bool(is_nonnegative),
        "is_integer": bool(is_integer),
        "inventory_feasible": bool(inventory_feasible),
        "demand_feasible": bool(demand_feasible),
        "level_feasible": bool(level_feasible),
        "reachability_feasible": bool(reachability_feasible),
        "zero_crossing": len(crossing_pairs) == 0,
        "objective": int(round(objective)) if is_integer and np.isfinite(objective) else objective,
        "negative_violation": negative_violation,
        "max_negative_violation": max_negative_violation,
        "integer_violation": integer_violation,
        "inventory_violation": inventory_violation,
        "max_inventory_violation": float(inventory_gap.max(initial=0.0)),
        "demand_violation": demand_violation,
        "max_demand_shortfall": float(demand_gap.max(initial=0.0)),
        "level_violation": level_violation,
        "max_level_violation": float(level_gap.max(initial=0.0)),
        "reachability_violation": unreachable_shots,
        "crossing_conflict_count": len(crossing_pairs),
        "crossing_pairs": [
            [[int(a[0]), int(a[1])], [int(b[0]), int(b[1])]]
            for a, b in crossing_pairs
        ],
        "active_routes": [[int(n), int(m)] for n, m in active_routes(A, tol=tol)],
        "effective_hits": effective_hits.tolist(),
        "demand_shortfall": demand_gap.tolist(),
        "tolerance": float(tol),
    }


def validate_solution(
    allocations: np.ndarray,
    scheme_idx: np.ndarray,
    B: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    G_th: np.ndarray,
    Q: np.ndarray,
    X: np.ndarray,
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    tol: float = 1e-7,
) -> dict[str, Any]:
    try:
        S_chosen, G_chosen = selected_plan(U, G, scheme_idx, tol=tol)
    except ValueError as exc:
        return {
            "is_feasible": False,
            "hard_feasible": False,
            "shape_valid": False,
            "scheme_valid": False,
            "error": str(exc),
        }

    report = validate_fixed_plan(
        allocations,
        B,
        S_chosen,
        G_chosen,
        G_th,
        Q,
        X,
        base_coords,
        target_coords,
        tol=tol,
    )
    report["scheme_valid"] = True
    report["scheme_idx"] = np.rint(np.asarray(scheme_idx, dtype=float)).astype(int).tolist()
    return report
