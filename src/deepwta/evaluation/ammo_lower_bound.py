from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TargetSchemeBound:
    target: int
    scheme: int
    legal: bool
    feasible: bool
    demand_total: float
    positive_weapon_count: int
    unlimited_shots: Optional[int]
    capacity_shots: Optional[int]
    local_inventory_penalty_shots: Optional[int]
    single_reachable_weapon_count: int
    min_reachable_base_count: int
    mean_best_survival: float


@dataclass(frozen=True)
class FractionalBoundResult:
    ammo_norm: float
    total_shots: int
    total_demand: float
    scheme_idx: np.ndarray
    iterations: int


def _ceil_required(demand: float, survival: float, tol: float) -> int:
    if demand <= tol:
        return 0
    if survival <= tol:
        raise ValueError("positive demand cannot be met with zero survival")
    return int(math.ceil(max(demand - tol, 0.0) / survival))


def _weapon_capacity_lower_bound(
    demand: float,
    survival: np.ndarray,
    capacity: np.ndarray,
    reachable: np.ndarray,
    tol: float,
) -> Optional[Tuple[int, int, int, float]]:
    """Return unlimited LB, capacity-aware LB, reachable count, best survival.

    The capacity-aware value is exact for a single (weapon,target) pair when the
    whole inventory vector is made available to that target. Greedy highest-
    survival use is optimal because every shot has equal unit cost.
    """
    demand = max(float(demand), 0.0)
    if demand <= tol:
        return 0, 0, int(np.sum(reachable)), 0.0

    survival = np.asarray(survival, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.int64)
    reachable = np.asarray(reachable, dtype=bool)
    valid = np.flatnonzero(reachable & (survival > tol) & (capacity > 0))
    if valid.size == 0:
        return None

    best_survival = float(np.max(survival[valid]))
    unlimited = _ceil_required(demand, best_survival, tol)

    order = sorted(valid.tolist(), key=lambda i: (-float(survival[i]), int(i)))
    remaining = demand
    shots = 0
    for i in order:
        s = float(survival[i])
        cap = int(capacity[i])
        need = _ceil_required(remaining, s, tol)
        take = min(cap, need)
        if take <= 0:
            continue
        shots += take
        remaining -= take * s
        if remaining <= tol:
            break

    if remaining > tol:
        return None
    return int(unlimited), int(shots), int(valid.size), best_survival


def build_target_scheme_bounds(
    B: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    G_th: np.ndarray,
    Q: np.ndarray,
    X: np.ndarray,
    tol: float = 1e-7,
) -> List[List[TargetSchemeBound]]:
    """Build per-target/per-scheme optimistic ammunition lower bounds.

    Two relaxations are reported:
      * unlimited_shots: each weapon may use its best reachable survival route
        with unlimited capacity;
      * capacity_shots: each target may independently use the full inventory B.

    Both ignore cross-target inventory coupling and crossing, therefore they are
    valid optimistic lower bounds for the full integer WTA problem.
    """
    B = np.asarray(B, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    G_th = np.asarray(G_th, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    L, K, M = U.shape
    if Q.shape[0] != K or Q.shape[2] != M:
        raise ValueError("U/Q dimensions are inconsistent")
    if B.shape != Q.shape[:2]:
        raise ValueError("B must have shape [K,N]")
    if G.shape != (L, M) or G_th.shape != (M,):
        raise ValueError("G/G_th dimensions are inconsistent")

    capacity = np.floor(np.maximum(B, 0.0) + tol).astype(np.int64)
    survival = np.maximum(1.0 - Q, 0.0)

    table: List[List[TargetSchemeBound]] = []
    for m in range(M):
        row: List[TargetSchemeBound] = []
        for l in range(L):
            legal = bool(G[l, m] >= G_th[m] - tol)
            demand_total = float(np.sum(np.maximum(U[l, :, m], 0.0)))
            if not legal:
                row.append(
                    TargetSchemeBound(
                        m, l, False, False, demand_total, 0, None, None, None,
                        0, 0, 0.0,
                    )
                )
                continue

            unlimited_total = 0
            capacity_total = 0
            positive_weapons = 0
            single_reachable = 0
            min_reachable = None
            best_survivals: List[float] = []
            feasible = True

            for k in range(K):
                demand = max(float(U[l, k, m]), 0.0)
                if demand <= tol:
                    continue
                positive_weapons += 1
                reachable = X[k, :, m] > 0.5
                result = _weapon_capacity_lower_bound(
                    demand,
                    survival[k, :, m],
                    capacity[k, :],
                    reachable,
                    tol,
                )
                if result is None:
                    feasible = False
                    break
                unlimited, capacity_shots, reachable_count, best_survival = result
                unlimited_total += unlimited
                capacity_total += capacity_shots
                single_reachable += int(reachable_count == 1)
                min_reachable = (
                    reachable_count
                    if min_reachable is None
                    else min(min_reachable, reachable_count)
                )
                best_survivals.append(best_survival)

            if feasible:
                row.append(
                    TargetSchemeBound(
                        target=m,
                        scheme=l,
                        legal=True,
                        feasible=True,
                        demand_total=demand_total,
                        positive_weapon_count=positive_weapons,
                        unlimited_shots=int(unlimited_total),
                        capacity_shots=int(capacity_total),
                        local_inventory_penalty_shots=int(
                            capacity_total - unlimited_total
                        ),
                        single_reachable_weapon_count=int(single_reachable),
                        min_reachable_base_count=int(min_reachable or 0),
                        mean_best_survival=float(np.mean(best_survivals))
                        if best_survivals else 0.0,
                    )
                )
            else:
                row.append(
                    TargetSchemeBound(
                        m, l, True, False, demand_total, positive_weapons,
                        None, None, None, single_reachable,
                        int(min_reachable or 0),
                        float(np.mean(best_survivals)) if best_survivals else 0.0,
                    )
                )
        table.append(row)
    return table


def _bound_arrays(
    table: List[List[TargetSchemeBound]],
    shot_field: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = len(table)
    L = len(table[0]) if M else 0
    shots = np.full((M, L), np.inf, dtype=np.float64)
    demand = np.zeros((M, L), dtype=np.float64)
    feasible = np.zeros((M, L), dtype=bool)
    for m, row in enumerate(table):
        for item in row:
            value = getattr(item, shot_field)
            demand[m, item.scheme] = item.demand_total
            if item.legal and item.feasible and value is not None:
                shots[m, item.scheme] = float(value)
                feasible[m, item.scheme] = True
    return shots, demand, feasible


def minimize_fractional_bound(
    table: List[List[TargetSchemeBound]],
    shot_field: str = "capacity_shots",
    tol: float = 1e-12,
    max_iter: int = 128,
) -> FractionalBoundResult:
    """Exact multiple-choice minimum of sum(shots)/sum(demand).

    Dinkelbach iterations are exact here because for fixed lambda the problem
    separates by target and each target has only a finite set of schemes.
    """
    shots, demand, feasible = _bound_arrays(table, shot_field)
    M, _ = shots.shape
    if M == 0:
        return FractionalBoundResult(0.0, 0, 0.0, np.zeros(0, dtype=np.int64), 0)
    if np.any(np.sum(feasible, axis=1) == 0):
        bad = np.flatnonzero(np.sum(feasible, axis=1) == 0).tolist()
        raise ValueError("no feasible legal scheme for targets: %s" % bad)

    # A deterministic feasible starting point.
    choices = np.empty(M, dtype=np.int64)
    for m in range(M):
        candidates = np.flatnonzero(feasible[m]).tolist()
        choices[m] = min(
            candidates,
            key=lambda l: (
                shots[m, l] / max(demand[m, l], tol),
                shots[m, l],
                -demand[m, l],
                int(l),
            ),
        )
    total_shots = float(np.sum(shots[np.arange(M), choices]))
    total_demand = float(np.sum(demand[np.arange(M), choices]))
    lam = total_shots / max(total_demand, tol)

    for iteration in range(1, max_iter + 1):
        new_choices = np.empty(M, dtype=np.int64)
        for m in range(M):
            candidates = np.flatnonzero(feasible[m]).tolist()
            new_choices[m] = min(
                candidates,
                key=lambda l: (
                    shots[m, l] - lam * demand[m, l],
                    shots[m, l],
                    -demand[m, l],
                    int(l),
                ),
            )
        new_shots = float(np.sum(shots[np.arange(M), new_choices]))
        new_demand = float(np.sum(demand[np.arange(M), new_choices]))
        new_lam = new_shots / max(new_demand, tol)
        residual = new_shots - lam * new_demand
        if np.array_equal(new_choices, choices) or abs(residual) <= tol:
            return FractionalBoundResult(
                ammo_norm=float(new_lam),
                total_shots=int(round(new_shots)),
                total_demand=float(new_demand),
                scheme_idx=new_choices.copy(),
                iterations=iteration,
            )
        choices = new_choices
        lam = new_lam

    raise RuntimeError("fractional lower-bound optimization did not converge")


def minimum_raw_shot_bound(
    table: List[List[TargetSchemeBound]],
    shot_field: str = "capacity_shots",
) -> Tuple[int, np.ndarray]:
    shots, demand, feasible = _bound_arrays(table, shot_field)
    M, _ = shots.shape
    choices = np.empty(M, dtype=np.int64)
    total = 0
    for m in range(M):
        candidates = np.flatnonzero(feasible[m]).tolist()
        if not candidates:
            raise ValueError("target %d has no feasible legal scheme" % m)
        l = min(
            candidates,
            key=lambda x: (
                shots[m, x],
                shots[m, x] / max(demand[m, x], 1e-12),
                -demand[m, x],
                int(x),
            ),
        )
        choices[m] = int(l)
        total += int(round(shots[m, l]))
    return int(total), choices


def selected_scheme_bound(
    table: List[List[TargetSchemeBound]],
    scheme_idx: np.ndarray,
    shot_field: str = "capacity_shots",
) -> Tuple[int, float, float]:
    scheme_idx = np.asarray(scheme_idx, dtype=np.int64).reshape(-1)
    if len(table) != scheme_idx.size:
        raise ValueError("scheme_idx length does not match target count")
    shots = 0
    demand = 0.0
    for m, l in enumerate(scheme_idx.tolist()):
        item = table[m][int(l)]
        value = getattr(item, shot_field)
        if not item.legal or not item.feasible or value is None:
            raise ValueError("selected scheme is infeasible in lower-bound audit")
        shots += int(value)
        demand += float(item.demand_total)
    return int(shots), float(demand), float(shots / max(demand, 1e-12))


def audit_solution_ammo(
    A: np.ndarray,
    scheme_idx: np.ndarray,
    table: List[List[TargetSchemeBound]],
    near_lb_ratio: float = 0.02,
) -> Tuple[dict, List[dict]]:
    A = np.asarray(A, dtype=np.int64)
    scheme_idx = np.asarray(scheme_idx, dtype=np.int64).reshape(-1)
    actual_ammo = int(np.sum(A))

    selected_cap_shots, selected_demand, selected_cap_norm = selected_scheme_bound(
        table, scheme_idx, "capacity_shots"
    )
    selected_unlim_shots, _, selected_unlim_norm = selected_scheme_bound(
        table, scheme_idx, "unlimited_shots"
    )
    global_cap = minimize_fractional_bound(table, "capacity_shots")
    global_unlim = minimize_fractional_bound(table, "unlimited_shots")
    raw_cap_shots, raw_cap_scheme = minimum_raw_shot_bound(table, "capacity_shots")
    raw_unlim_shots, raw_unlim_scheme = minimum_raw_shot_bound(
        table, "unlimited_shots"
    )

    actual_norm = float(actual_ammo / max(selected_demand, 1e-12))
    allocation_gap = max(actual_norm - selected_cap_norm, 0.0)
    scheme_gap = max(selected_cap_norm - global_cap.ammo_norm, 0.0)
    local_inventory_gap = max(global_cap.ammo_norm - global_unlim.ammo_norm, 0.0)
    total_relaxed_gap = max(actual_norm - global_unlim.ammo_norm, 0.0)
    gap_to_capacity_ratio = float(
        actual_norm / max(global_cap.ammo_norm, 1e-12) - 1.0
    )

    components = {
        "allocation_cross_target_gap": allocation_gap,
        "scheme_choice_gap": scheme_gap,
        "local_inventory_gap": local_inventory_gap,
    }
    if gap_to_capacity_ratio <= near_lb_ratio:
        bottleneck = "near_relaxed_lb"
    elif total_relaxed_gap <= 1e-12:
        bottleneck = "near_relaxed_lb"
    else:
        name, value = max(components.items(), key=lambda kv: kv[1])
        share = value / max(total_relaxed_gap, 1e-12)
        bottleneck = name if share >= 0.60 else "mixed"

    scenario = {
        "actual_ammo": actual_ammo,
        "actual_ammo_norm": actual_norm,
        "selected_demand_total": selected_demand,
        "selected_unlimited_lb_shots": selected_unlim_shots,
        "selected_capacity_lb_shots": selected_cap_shots,
        "selected_unlimited_lb_norm": selected_unlim_norm,
        "selected_capacity_lb_norm": selected_cap_norm,
        "global_unlimited_lb_shots_at_norm_opt": global_unlim.total_shots,
        "global_unlimited_lb_demand_at_norm_opt": global_unlim.total_demand,
        "global_unlimited_lb_norm": global_unlim.ammo_norm,
        "global_capacity_lb_shots_at_norm_opt": global_cap.total_shots,
        "global_capacity_lb_demand_at_norm_opt": global_cap.total_demand,
        "global_capacity_lb_norm": global_cap.ammo_norm,
        "global_raw_unlimited_lb_shots": raw_unlim_shots,
        "global_raw_capacity_lb_shots": raw_cap_shots,
        "actual_minus_selected_capacity_shots": actual_ammo - selected_cap_shots,
        "selected_minus_global_raw_capacity_shots": selected_cap_shots
        - raw_cap_shots,
        "raw_gap_to_global_capacity_lb_shots": actual_ammo - raw_cap_shots,
        "ammo_norm_allocation_cross_target_gap": allocation_gap,
        "ammo_norm_scheme_choice_gap": scheme_gap,
        "ammo_norm_local_inventory_gap": local_inventory_gap,
        "ammo_norm_total_gap_to_unlimited_lb": total_relaxed_gap,
        "ammo_norm_gap_to_capacity_lb_ratio": gap_to_capacity_ratio,
        "ammo_norm_gap_to_unlimited_lb_ratio": float(
            actual_norm / max(global_unlim.ammo_norm, 1e-12) - 1.0
        ),
        "near_capacity_lb": bool(gap_to_capacity_ratio <= near_lb_ratio),
        "bottleneck_class": bottleneck,
        "capacity_norm_lb_iterations": global_cap.iterations,
        "unlimited_norm_lb_iterations": global_unlim.iterations,
    }

    target_rows: List[dict] = []
    for m, selected in enumerate(scheme_idx.tolist()):
        selected_item = table[m][int(selected)]
        feasible_items = [x for x in table[m] if x.legal and x.feasible]
        best_raw = min(
            feasible_items,
            key=lambda x: (
                int(x.capacity_shots),
                int(x.unlimited_shots),
                x.scheme,
            ),
        )
        actual_target_ammo = int(np.sum(A[:, :, m]))
        norm_scheme = int(global_cap.scheme_idx[m])
        norm_item = table[m][norm_scheme]
        target_rows.append(
            {
                "target": int(m),
                "selected_scheme": int(selected),
                "actual_target_ammo": actual_target_ammo,
                "selected_demand_total": float(selected_item.demand_total),
                "selected_unlimited_lb_shots": int(selected_item.unlimited_shots),
                "selected_capacity_lb_shots": int(selected_item.capacity_shots),
                "target_allocation_gap_shots": actual_target_ammo
                - int(selected_item.capacity_shots),
                "selected_local_inventory_penalty_shots": int(
                    selected_item.local_inventory_penalty_shots
                ),
                "selected_single_reachable_weapon_count": int(
                    selected_item.single_reachable_weapon_count
                ),
                "selected_min_reachable_base_count": int(
                    selected_item.min_reachable_base_count
                ),
                "selected_mean_best_survival": float(
                    selected_item.mean_best_survival
                ),
                "legal_feasible_scheme_count": int(len(feasible_items)),
                "best_raw_capacity_scheme": int(best_raw.scheme),
                "best_raw_capacity_lb_shots": int(best_raw.capacity_shots),
                "raw_scheme_potential_shots": int(selected_item.capacity_shots)
                - int(best_raw.capacity_shots),
                "best_norm_capacity_scheme": norm_scheme,
                "best_norm_capacity_lb_shots": int(norm_item.capacity_shots),
                "best_norm_capacity_demand": float(norm_item.demand_total),
                "selected_matches_norm_lb_scheme": bool(selected == norm_scheme),
                "selected_matches_raw_lb_scheme": bool(selected == best_raw.scheme),
                "raw_unlimited_best_scheme": int(raw_unlim_scheme[m]),
                "raw_capacity_best_scheme": int(raw_cap_scheme[m]),
            }
        )
    return scenario, target_rows


def scheme_table_rows(
    table: List[List[TargetSchemeBound]],
) -> List[dict]:
    rows: List[dict] = []
    for target_row in table:
        for item in target_row:
            if not item.legal:
                continue
            rows.append(
                {
                    "target": item.target,
                    "scheme": item.scheme,
                    "legal": item.legal,
                    "feasible": item.feasible,
                    "demand_total": item.demand_total,
                    "positive_weapon_count": item.positive_weapon_count,
                    "unlimited_lb_shots": item.unlimited_shots,
                    "capacity_lb_shots": item.capacity_shots,
                    "local_inventory_penalty_shots": item.local_inventory_penalty_shots,
                    "single_reachable_weapon_count": item.single_reachable_weapon_count,
                    "min_reachable_base_count": item.min_reachable_base_count,
                    "mean_best_survival": item.mean_best_survival,
                }
            )
    return rows
