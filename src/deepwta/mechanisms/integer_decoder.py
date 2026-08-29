from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict

import numpy as np




@dataclass
class CandidateList:
    ordered_bases: np.ndarray
    active_count: int

    def current(self) -> np.ndarray:
        return self.ordered_bases[: self.active_count]

    def all(self) -> np.ndarray:
        return self.ordered_bases

    def expand(self) -> bool:
        if self.active_count >= len(self.ordered_bases):
            return False
        self.active_count = min(
            len(self.ordered_bases),
            max(self.active_count + 1, self.active_count * 2),
        )
        return True

    @property
    def exhausted(self) -> bool:
        return self.active_count >= len(self.ordered_bases)


def _stable_softmax(values: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    if not np.all(np.isfinite(x)):
        raise ValueError("candidate scores must be finite")
    if tau <= 0.0:
        raise ValueError("tau must be positive")

    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo <= 1e-12:
        return np.full(x.shape, 1.0 / x.size, dtype=np.float64)

    # Scale to [0,1] before temperature so tau behaves consistently across
    # graph sizes and absolute allocation magnitudes.
    x = (x - lo) / (hi - lo)
    logits = x / tau
    logits -= float(np.max(logits))
    exp = np.exp(np.clip(logits, -60.0, 60.0))
    den = float(np.sum(exp))
    if den <= 0.0:
        return np.full(x.shape, 1.0 / x.size, dtype=np.float64)
    return exp / den


def adaptive_k_from_scores(
    scores: np.ndarray,
    rho: float,
    k_min: int,
    k_max: int,
    tau: float,
) -> tuple[np.ndarray, int]:
    scores = np.asarray(scores, dtype=np.float64)
    n = int(scores.size)
    if n == 0:
        return np.empty(0, dtype=np.int64), 0

    probs = _stable_softmax(scores, tau=tau)
    order = np.argsort(-probs, kind="stable").astype(np.int64)
    cumulative = np.cumsum(probs[order])
    reached = int(np.searchsorted(cumulative, rho, side="left") + 1)
    k = max(int(k_min), reached)
    k = min(k, int(k_max), n)
    return order, k


def floor_integerize(
    assign_intensity: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
    *,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Element-wise floor with exact physical masking and inventory safety.

    DeepWTA uses floor as a conservative integer seed, but no longer assumes
    that floor alone preserves the learned route topology. Fractional structure
    is recovered at the route level by the target-concentrated decoder.
    """
    soft = np.asarray(assign_intensity, dtype=np.float64)
    reachable = np.asarray(X, dtype=np.float64) > 0.5
    inventory = np.maximum(
        np.floor(np.asarray(B, dtype=np.float64) + tol),
        0.0,
    ).astype(np.int64)

    if soft.ndim != 3 or soft.shape != reachable.shape:
        raise ValueError("assign_intensity and X must have shape [K,N,M]")
    K, N, M = soft.shape
    if inventory.shape != (K, N):
        raise ValueError("B must have shape [K,N]")

    clean = np.maximum(soft, 0.0) * reachable
    floored = np.floor(clean + 1e-12).astype(np.int64)
    residual = clean - floored.astype(np.float64)

    # A post-projection DeepWTA output should make this path unnecessary, but
    # retain deterministic defensive clipping for legacy/numeric inputs.
    clipped_shots = 0
    clipped_entries = 0
    for k in range(K):
        for i in range(N):
            overflow = int(np.sum(floored[k, i, :])) - int(inventory[k, i])
            if overflow <= 0:
                continue

            targets = np.flatnonzero(floored[k, i, :] > 0)
            targets = sorted(
                targets.tolist(),
                key=lambda m: (
                    float(clean[k, i, m]),
                    float(residual[k, i, m]),
                    int(m),
                ),
            )
            for m in targets:
                if overflow <= 0:
                    break
                remove = min(int(floored[k, i, m]), overflow)
                if remove <= 0:
                    continue
                floored[k, i, m] -= remove
                overflow -= remove
                clipped_shots += remove
                clipped_entries += 1

            if overflow > 0:
                raise RuntimeError("defensive floor inventory clipping failed")

    return floored, residual, {
        "floor_inventory_clipped_shots": int(clipped_shots),
        "floor_inventory_clipped_entries": int(clipped_entries),
    }


def residual_aware_integerize(
    assign_intensity: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
    selected_demand: np.ndarray,
    Q: np.ndarray,
    *,
    mode: str = "floor",
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a deterministic, inventory-safe integer seed.

    Each positive fractional entry may contribute at most one round-up shot.
    Round-up is restricted to floor-shortage weapon-target pairs and physical
    routes.  Ranking is stable in target order, so tied scores are reproducible.
    """
    modes = {"floor", "ra_r", "ra_s", "ra_n", "gb_r", "gb_s", "gb_sr"}
    if mode not in modes:
        raise ValueError(f"unsupported integer seed mode: {mode}")

    soft = np.asarray(assign_intensity, dtype=np.float64)
    reachable = np.asarray(X, dtype=np.float64) > 0.5
    inventory = np.maximum(
        np.floor(np.asarray(B, dtype=np.float64) + tol), 0.0
    ).astype(np.int64)
    demand = np.maximum(np.asarray(selected_demand, dtype=np.float64), 0.0)
    q = np.asarray(Q, dtype=np.float64)
    if soft.ndim != 3 or soft.shape != reachable.shape or q.shape != soft.shape:
        raise ValueError("assign_intensity, X and Q must have shape [K,N,M]")
    K, N, M = soft.shape
    if inventory.shape != (K, N):
        raise ValueError("B must have shape [K,N]")
    if demand.shape != (K, M):
        raise ValueError("selected_demand must have shape [K,M]")
    if not np.all(np.isfinite(soft)) or not np.all(np.isfinite(q)):
        raise ValueError("assign_intensity and Q must be finite")

    floor_seed, residual, floor_report = floor_integerize(
        soft, reachable, inventory, tol=tol
    )
    survival = np.clip(1.0 - q, 0.0, 1.0) * reachable
    floor_arrivals = np.sum(survival * floor_seed, axis=1)
    floor_shortage = np.maximum(demand - floor_arrivals, 0.0)

    seed = floor_seed.copy()
    rounding_started = time.perf_counter()
    global_mode = mode in {"gb_r", "gb_s", "gb_sr"}
    global_report = {
        "integer_seed_tentative_roundups": 0.0,
        "integer_seed_base_pruned_roundups": 0.0,
        "integer_seed_reconciled_roundups": 0.0,
    }
    if global_mode:
        remaining = np.maximum(
            inventory - np.sum(floor_seed, axis=2), 0
        ).astype(np.int64)
        floor_route_active = np.sum(floor_seed, axis=0) > 0
        eligible = (
            reachable
            & (residual > tol)
            & (survival > tol)
            & (floor_shortage[:, None, :] > tol)
            & (remaining[:, :, None] > 0)
        )
        marginal = np.minimum(floor_shortage[:, None, :], survival)
        score = residual if mode == "gb_r" else residual * marginal

        def ordered_indices(k: int, j: int, bases: np.ndarray) -> np.ndarray:
            r = residual[k, bases, j]
            s = survival[k, bases, j]
            d = marginal[k, bases, j]
            if mode == "gb_r":
                order = np.lexsort((bases, -s, -d, -r))
            elif mode == "gb_s":
                order = np.lexsort((bases, -s, -r, -d, -score[k, bases, j]))
            else:
                active = floor_route_active[bases, j].astype(np.int8)
                order = np.lexsort((bases, -s, -r, -d, -active))
            return bases[order]

        base_index = np.broadcast_to(
            np.arange(N, dtype=np.int64)[None, :, None], (K, N, M)
        )
        active_key = np.broadcast_to(
            floor_route_active[None, :, :], (K, N, M)
        ).astype(np.int8)
        eligible_key = eligible.astype(np.int8)
        if mode == "gb_r":
            base_order = np.lexsort(
                (base_index, -survival, -marginal, -residual, -eligible_key), axis=1
            )
        elif mode == "gb_s":
            base_order = np.lexsort(
                (
                    base_index, -survival, -residual, -marginal,
                    -score, -eligible_key,
                ),
                axis=1,
            )
        else:
            base_order = np.lexsort(
                (
                    base_index, -survival, -residual, -marginal,
                    -active_key, -eligible_key,
                ),
                axis=1,
            )
        ordered_survival = np.take_along_axis(survival, base_order, axis=1)
        ordered_eligible = np.take_along_axis(eligible, base_order, axis=1)
        cumulative = np.cumsum(ordered_survival * ordered_eligible, axis=1)
        previous = cumulative - ordered_survival * ordered_eligible
        selected_ordered = ordered_eligible & (
            previous < floor_shortage[:, None, :] - tol
        )
        tentative = np.zeros_like(eligible)
        np.put_along_axis(tentative, base_order, selected_ordered, axis=1)

        target_index = np.broadcast_to(
            np.arange(M, dtype=np.int64)[None, None, :], (K, N, M)
        )
        tentative_key = tentative.astype(np.int8)
        if mode == "gb_r":
            target_order = np.lexsort(
                (target_index, -survival, -marginal, -residual, -tentative_key),
                axis=2,
            )
        elif mode == "gb_s":
            target_order = np.lexsort(
                (
                    target_index, -survival, -residual, -marginal,
                    -score, -tentative_key,
                ),
                axis=2,
            )
        else:
            target_order = np.lexsort(
                (
                    target_index, -survival, -residual, -marginal,
                    -active_key, -tentative_key,
                ),
                axis=2,
            )
        ordered_tentative = np.take_along_axis(tentative, target_order, axis=2)
        target_position = np.broadcast_to(
            np.arange(M, dtype=np.int64)[None, None, :], (K, N, M)
        )
        keep_ordered = ordered_tentative & (
            target_position < remaining[:, :, None]
        )
        accepted = np.zeros_like(tentative)
        np.put_along_axis(accepted, target_order, keep_ordered, axis=2)
        pruned = int(np.sum(tentative) - np.sum(accepted))

        # One deterministic target-side reconciliation pass. Capacity is
        # consumed immediately, so this pass cannot create a second conflict.
        current_arrivals = floor_arrivals + np.sum(
            survival * accepted, axis=1
        )
        reconcile_shortage = np.maximum(demand - current_arrivals, 0.0)
        free = remaining - np.sum(accepted, axis=2).astype(np.int64)
        reconciled = 0
        for k, j in np.argwhere(reconcile_shortage > tol).tolist():
            need = float(reconcile_shortage[k, j])
            bases = np.flatnonzero(
                eligible[k, :, j]
                & (~accepted[k, :, j])
                & (free[k, :] > 0)
            ).astype(np.int64)
            if bases.size == 0:
                continue
            ordered = ordered_indices(k, j, bases)
            cumulative = np.cumsum(survival[k, ordered, j])
            count = (
                int(np.searchsorted(cumulative, need - tol, side="left") + 1)
                if cumulative[-1] >= need - tol
                else int(ordered.size)
            )
            for i in ordered[:count].tolist():
                if free[k, i] <= 0:
                    continue
                accepted[k, i, j] = True
                free[k, i] -= 1
                reconciled += 1

        seed += accepted.astype(np.int64)
        global_report = {
            "integer_seed_tentative_roundups": float(np.sum(tentative)),
            "integer_seed_base_pruned_roundups": float(pruned),
            "integer_seed_reconciled_roundups": float(reconciled),
        }
    elif mode != "floor":
        eligible = (
            reachable
            & (residual > tol)
            & (floor_shortage[:, None, :] > tol)
        )
        marginal = np.minimum(
            floor_shortage[:, None, :], survival
        )
        if mode == "ra_r":
            score = residual
        elif mode == "ra_s":
            score = residual * marginal
        else:
            score = residual * marginal / np.maximum(
                floor_shortage[:, None, :], tol
            )
        score = np.where(eligible, score, 0.0)

        remaining = np.maximum(
            inventory - np.sum(floor_seed, axis=2), 0
        ).astype(np.int64)
        # Stable descending ranks along target dimension implement variable
        # Top-m for all (weapon, base) pairs without Python loops.
        order = np.argsort(-score, axis=2, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(
            ranks,
            order,
            np.broadcast_to(np.arange(M), (K, N, M)),
            axis=2,
        )
        selected = (score > 0.0) & (ranks < remaining[:, :, None])
        seed += selected.astype(np.int64)
    rounding_runtime = (
        time.perf_counter() - rounding_started if mode != "floor" else 0.0
    )

    seed_arrivals = np.sum(survival * seed, axis=1)
    seed_shortage = np.maximum(demand - seed_arrivals, 0.0)
    demand_total = max(float(np.sum(demand)), tol)
    soft_total = max(float(np.sum(np.maximum(soft, 0.0) * reachable)), tol)
    effective_soft = max(
        float(np.sum(survival * np.maximum(soft, 0.0))), tol
    )
    roundup_shots = int(np.sum(seed - floor_seed))
    floor_shortage_total = float(np.sum(floor_shortage))
    seed_shortage_total = float(np.sum(seed_shortage))
    floor_routes = np.sum(floor_seed, axis=0) > 0
    seed_routes = np.sum(seed, axis=0) > 0
    inventory_excess = np.maximum(
        np.sum(seed, axis=2) - inventory, 0
    )
    unreachable = seed * (~reachable)
    global_oversupply = float(np.sum(np.maximum(seed_arrivals - demand, 0.0)))

    report = {
        **floor_report,
        "integer_seed_mode": str(mode),
        "integer_seed_sum": float(np.sum(seed)),
        "integer_floor_sum": float(np.sum(floor_seed)),
        "integer_seed_effective_arrivals": float(np.sum(seed_arrivals)),
        "integer_floor_effective_arrivals": float(np.sum(floor_arrivals)),
        "integer_seed_shortage": seed_shortage_total / demand_total,
        "integer_floor_shortage": floor_shortage_total / demand_total,
        "integer_seed_shortage_absolute": seed_shortage_total,
        "integer_floor_shortage_absolute": floor_shortage_total,
        "integer_seed_targets_needing_repair": float(
            np.sum(np.any(seed_shortage > tol, axis=0))
        ),
        "integer_seed_shortage_pairs": float(np.sum(seed_shortage > tol)),
        "integer_seed_inventory_violation": float(np.sum(inventory_excess)),
        "integer_seed_reachability_violation": float(np.sum(unreachable)),
        "integer_seed_retention": float(np.sum(seed)) / soft_total,
        "integer_floor_retention": float(np.sum(floor_seed)) / soft_total,
        "integer_seed_effective_retention": float(np.sum(seed_arrivals)) / effective_soft,
        "integer_floor_effective_retention": float(np.sum(floor_arrivals)) / effective_soft,
        "integer_seed_oversupplied_effective_demand": global_oversupply,
        "integer_seed_oversupply_ratio": float(
            global_oversupply / max(floor_shortage_total, tol)
        ),
        "integer_seed_roundup_shots": float(roundup_shots),
        "integer_seed_rounding_efficiency": float(
            (floor_shortage_total - seed_shortage_total) / max(roundup_shots, tol)
        ),
        "integer_floor_active_routes": float(np.sum(floor_routes)),
        "integer_seed_active_routes": float(np.sum(seed_routes)),
        "integer_seed_new_routes": float(np.sum(seed_routes & ~floor_routes)),
        "integer_seed_rounding_kernel_s": float(rounding_runtime),
        **global_report,
    }
    return seed, floor_seed, residual, report


def soft_route_mass(assign_intensity: np.ndarray) -> np.ndarray:
    """Return route-level soft mass [N,M] = sum_k A_soft[k,i,j]."""
    soft = np.maximum(np.asarray(assign_intensity, dtype=np.float64), 0.0)
    if soft.ndim != 3:
        raise ValueError("assign_intensity must have shape [K,N,M]")
    return np.sum(soft, axis=0)


def residual_route_mass(residual: np.ndarray) -> np.ndarray:
    residual = np.maximum(np.asarray(residual, dtype=np.float64), 0.0)
    if residual.ndim != 3:
        raise ValueError("residual must have shape [K,N,M]")
    return np.sum(residual, axis=0)


def build_target_route_candidates(
    assign_intensity: np.ndarray,
    X: np.ndarray,
    *,
    use_topk: bool,
    rho: float = 0.90,
    k_min: int = 2,
    k_max: int = 8,
    tau: float = 0.25,
) -> Dict[int, CandidateList]:
    """Build one ordered base-route list per target.

    The ranking unit is route (base,target). A base is eligible when at least
    one weapon can reach the target. The learned score is total soft route mass
    across weapons; this directly matches the route-level crossing topology.
    """
    soft = np.asarray(assign_intensity, dtype=np.float64)
    reachable = np.asarray(X, dtype=np.float64) > 0.5
    if soft.ndim != 3 or soft.shape != reachable.shape:
        raise ValueError("assign_intensity and X must have shape [K,N,M]")

    _, N, M = soft.shape
    route_mass = soft_route_mass(soft)
    result: Dict[int, CandidateList] = {}

    for m in range(M):
        bases = np.flatnonzero(np.any(reachable[:, :, m], axis=0)).astype(np.int64)
        if bases.size == 0:
            result[int(m)] = CandidateList(
                ordered_bases=np.empty(0, dtype=np.int64),
                active_count=0,
            )
            continue

        values = route_mass[bases, m]
        if use_topk:
            local_order, active_count = adaptive_k_from_scores(
                values,
                rho=rho,
                k_min=k_min,
                k_max=k_max,
                tau=tau,
            )
            ordered = bases[local_order]
        else:
            ordered = bases[np.argsort(-values, kind="stable")]
            active_count = len(ordered)

        result[int(m)] = CandidateList(
            ordered_bases=np.asarray(ordered, dtype=np.int64),
            active_count=int(active_count),
        )

    return result


def target_route_union(
    candidates: Dict[int, CandidateList],
    *,
    current_only: bool = True,
) -> list[tuple[int, int]]:
    routes: list[tuple[int, int]] = []
    for m, item in candidates.items():
        bases = item.current() if current_only else item.all()
        routes.extend((int(i), int(m)) for i in bases.tolist())
    return sorted(set(routes))
