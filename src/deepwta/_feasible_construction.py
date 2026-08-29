from __future__ import annotations

from itertools import product
import time

import numpy as np

from deepwta.utils.geometry import (
    active_routes,
    crossing_pairs_from_allocations,
    find_crossing_pairs,
    route_conflict,
)
from deepwta.utils.validator import selected_plan, validate_solution


CONSTRUCTION_ALGORITHM = "deepwta-feasible-construction"

# Same soft-objective weights used by the corrected evaluator.
_W_AMMO = 10.0
_W_CROSS = 80.0
_W_FLIGHT = 30.0
_W_INTERCEPT = 10.0
_GNN_ROUTE_BIAS = 2.0

_LOCAL_TARGET_LIMIT = 10
_LOCAL_PAIR_LIMIT = 16
_LOCAL_SCHEME_TOPK = 3
_MIN_SOFT_GAIN = 1e-8


def construct_feasible_solution(
    assign_intensity,
    scheme_probs,
    B,
    U,
    G,
    G_th,
    Q,
    X,
    base_coords,
    target_coords,
    max_passes=None,
    tol=1e-7,
    return_report=False,
    flight_times=None,
    initial_allocation=None,
    initial_floor_allocation=None,
    initial_scheme_idx=None,
):
    repair_total_start = time.perf_counter()

    assign_intensity = np.asarray(assign_intensity, dtype=float)
    scheme_probs = np.asarray(scheme_probs, dtype=float)
    B = np.asarray(B, dtype=float)
    U = np.asarray(U, dtype=float)
    G = np.asarray(G, dtype=float)
    G_th = np.asarray(G_th, dtype=float)
    Q = np.asarray(Q, dtype=float)
    X = np.asarray(X, dtype=float)
    base_coords = np.asarray(base_coords, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)

    if Q.ndim != 3:
        raise ValueError("Q must have shape [K, N, M]")
    K, N, M = Q.shape
    if U.ndim != 3:
        raise ValueError("U must have shape [L, K, M]")
    L = U.shape[0]

    expected_shapes = {
        "assign_intensity": (K, N, M),
        "scheme_probs": (M, L),
        "B": (K, N),
        "U": (L, K, M),
        "G": (L, M),
        "G_th": (M,),
        "X": (K, N, M),
    }
    actual_shapes = {
        "assign_intensity": assign_intensity.shape,
        "scheme_probs": scheme_probs.shape,
        "B": B.shape,
        "U": U.shape,
        "G": G.shape,
        "G_th": G_th.shape,
        "X": X.shape,
    }
    if expected_shapes != actual_shapes:
        raise ValueError(f"shape mismatch: expected {expected_shapes}, got {actual_shapes}")
    if base_coords.shape != (N, 2) or target_coords.shape != (M, 2):
        raise ValueError("base_coords/target_coords must have shapes [N,2]/[M,2]")
    if not all(
        np.all(np.isfinite(arr))
        for arr in (assign_intensity, scheme_probs, B, U, G, G_th, Q, X)
    ):
        raise ValueError("repair inputs must be finite")

    seed_preserving_requested = initial_allocation is not None
    if seed_preserving_requested:
        initial_allocation = np.asarray(initial_allocation, dtype=float)
        initial_floor_allocation = np.asarray(
            initial_floor_allocation
            if initial_floor_allocation is not None
            else np.floor(initial_allocation + tol),
            dtype=float,
        )
        initial_scheme_idx = np.asarray(initial_scheme_idx, dtype=np.int64)
        if initial_allocation.shape != (K, N, M):
            raise ValueError("initial_allocation must have shape [K,N,M]")
        if initial_floor_allocation.shape != (K, N, M):
            raise ValueError("initial_floor_allocation must have shape [K,N,M]")
        if initial_scheme_idx.shape != (M,):
            raise ValueError("initial_scheme_idx must have shape [M]")
        if np.any(initial_scheme_idx < 0) or np.any(initial_scheme_idx >= L):
            raise ValueError("initial_scheme_idx contains invalid indices")

    inventory_cap = np.maximum(np.floor(B + tol), 0.0).astype(np.int64)
    raw_scheme_idx = np.argmax(scheme_probs, axis=1).astype(np.int64)
    scheme_has_level = np.ones(M, dtype=bool)

    route_distance = np.linalg.norm(
        base_coords[:, None, :] - target_coords[None, :, :], axis=2
    )
    max_distance = max(float(np.max(route_distance)), tol)
    fallback_time = route_distance / max_distance

    if flight_times is None:
        T_metric = np.broadcast_to(fallback_time[None, :, :], (K, N, M)).copy()
    else:
        T = np.asarray(flight_times, dtype=float)
        if T.shape == (N, M):
            T_metric = np.broadcast_to(T[None, :, :], (K, N, M)).copy()
        elif T.shape == (1, N, M):
            T_metric = np.broadcast_to(T, (K, N, M)).copy()
        elif T.shape == (K, N, M):
            T_metric = T.copy()
        else:
            raise ValueError(
                "flight_times must have shape [N,M], [1,N,M], or [K,N,M]"
            )
        if not np.all(np.isfinite(T_metric)):
            raise ValueError("flight_times must be finite")

    cross_base = (M * M) / (4.0 if M <= 20 else (6.0 if M <= 40 else 8.0))
    cross_base = max(float(cross_base), 1.0)

    A = np.zeros((K, N, M), dtype=np.int64)
    remain_B = inventory_cap.copy()
    scheme_idx = np.zeros(M, dtype=np.int64)

    conflict_cache: dict[tuple[tuple[int, int], tuple[int, int]], bool] = {}

    def cached_route_conflict(route_a, route_b):
        a = (int(route_a[0]), int(route_a[1]))
        b = (int(route_b[0]), int(route_b[1]))
        key = (a, b) if a <= b else (b, a)
        value = conflict_cache.get(key)
        if value is None:
            value = bool(route_conflict(base_coords, target_coords, a, b))
            conflict_cache[key] = value
        return value

    def valid_schemes(m):
        valid = np.flatnonzero(G[:, m] >= G_th[m] - tol)
        if valid.size == 0:
            scheme_has_level[m] = False
        return valid

    def scheme_order(m):
        valid = valid_schemes(m)
        if valid.size == 0:
            return [int(np.argmax(G[:, m]))]

        preferred = int(raw_scheme_idx[m])
        ordered = sorted(
            valid.tolist(),
            key=lambda l: (
                -float(scheme_probs[m, l]),
                float(np.sum(U[l, :, m])),
                int(l),
            ),
        )
        if preferred in ordered:
            ordered.remove(preferred)
            ordered.insert(0, preferred)
        return [int(x) for x in ordered]

    scheme_orders = [scheme_order(m) for m in range(M)]

    seed_trimmed_shots = 0
    seed_template = None
    seed_choices = None
    if seed_preserving_requested:
        seed_template = np.maximum(np.floor(initial_allocation + tol), 0.0).astype(
            np.int64
        )
        floor_reference = np.maximum(
            np.floor(initial_floor_allocation + tol), 0.0
        ).astype(np.int64)
        seed_template *= (X > 0.5)
        floor_reference *= (X > 0.5)
        if np.any(np.sum(seed_template, axis=2) > inventory_cap):
            raise ValueError("initial_allocation exceeds integer inventory")
        seed_choices = initial_scheme_idx.copy()
        selected_seed_demand, _ = selected_plan(U, G, seed_choices, tol=tol)
        floor_routes = np.sum(floor_reference, axis=0) > 0
        residual_signal = np.maximum(assign_intensity - floor_reference, 0.0)
        survival = np.clip(1.0 - Q, 0.0, 1.0)
        for k in range(K):
            for m in range(M):
                delivered = float(np.sum(survival[k, :, m] * seed_template[k, :, m]))
                demand_value = float(selected_seed_demand[k, m])
                if delivered <= demand_value + tol:
                    continue
                candidates = np.flatnonzero(
                    seed_template[k, :, m] > floor_reference[k, :, m]
                ).tolist()
                candidates.sort(
                    key=lambda n: (
                        0 if not floor_routes[int(n), m] else 1,
                        float(residual_signal[k, int(n), m]),
                        -float(Q[k, int(n), m]),
                        -float(T_metric[k, int(n), m]),
                        int(n),
                    )
                )
                for n in candidates:
                    hit = float(survival[k, n, m])
                    if delivered - hit < demand_value - tol:
                        continue
                    seed_template[k, n, m] -= 1
                    delivered -= hit
                    seed_trimmed_shots += 1

    def preferred_scheme(m):
        return int(scheme_orders[m][0])

    def target_priority(m, inventory):
        scheme = preferred_scheme(m)
        ratios = []
        reachable_counts = []
        for k in range(K):
            need = float(U[scheme, k, m])
            if need <= tol:
                continue
            mask = (
                (X[k, :, m] > 0.5)
                & (inventory[k] > 0)
                & ((1.0 - Q[k, :, m]) > tol)
            )
            reachable_counts.append(int(mask.sum()))
            capacity = float(
                np.sum(inventory[k, mask] * np.maximum(1.0 - Q[k, mask, m], 0.0))
            )
            ratios.append(capacity / max(need, tol))

        if not ratios:
            return (float("inf"), float("inf"), int(m))
        return (min(ratios), min(reachable_counts), int(m))

    scarcity_order = sorted(range(M), key=lambda m: target_priority(m, inventory_cap))

    def target_cross_scores(matrix, m):
        existing = [route for route in active_routes(matrix, tol=tol) if route[1] != m]
        scores = np.zeros(N, dtype=np.int64)
        if not existing:
            return scores

        candidates = [(n, m) for n in range(N)]
        existing_set = set(existing)
        candidate_set = set(candidates)
        for route_a, route_b in find_crossing_pairs(
            base_coords, target_coords, existing + candidates
        ):
            if route_a in candidate_set and route_b in existing_set:
                scores[route_a[0]] += 1
            elif route_b in candidate_set and route_a in existing_set:
                scores[route_b[0]] += 1
        return scores

    def weapon_priority(k, m, scheme, inventory):
        need = float(U[scheme, k, m])
        if need <= tol:
            return (float("inf"), float("inf"), int(k))
        mask = (
            (X[k, :, m] > 0.5)
            & (inventory[k] > 0)
            & ((1.0 - Q[k, :, m]) > tol)
        )
        count = int(mask.sum())
        capacity = float(
            np.sum(inventory[k, mask] * np.maximum(1.0 - Q[k, mask, m], 0.0))
        )
        return (capacity / max(need, tol), count, int(k))

    def route_rank_score(k, n, m, cross_scores, valid_bases, remaining_need):
        intensity_values = assign_intensity[k, valid_bases, m]
        lo = float(np.min(intensity_values)) if len(intensity_values) else 0.0
        hi = float(np.max(intensity_values)) if len(intensity_values) else 0.0
        if hi - lo <= tol:
            intensity_norm = 0.0
        else:
            intensity_norm = (float(assign_intensity[k, n, m]) - lo) / (hi - lo)

        p_hit = max(1.0 - float(Q[k, n, m]), tol)
        expected_shots = min(
            int(np.ceil(max(remaining_need - tol, 0.0) / p_hit)),
            int(max(inventory_cap[k, n], 1)),
        )
        time_value = float(T_metric[k, n, m])
        cross_term = float(cross_scores[n]) / cross_base
        ammo_term = float(expected_shots) / max(float(np.sum(U)), 1.0)

        return (
            _W_AMMO * ammo_term
            + _W_CROSS * cross_term
            + _W_FLIGHT * time_value
            + _W_INTERCEPT * float(Q[k, n, m])
            - _GNN_ROUTE_BIAS * intensity_norm
        )

    def greedy_allocate(need, k, m, inventory, matrix, cross_scores, mode="balanced"):
        remaining = float(need)
        if remaining <= tol:
            return True

        valid = np.flatnonzero(
            (X[k, :, m] > 0.5)
            & (inventory[k] > 0)
            & ((1.0 - Q[k, :, m]) > tol)
        )
        if valid.size == 0:
            return False

        valid_list = valid.tolist()

        def key(n):
            reuse = 0 if np.any(matrix[:, n, m] > 0) else 1
            soft = route_rank_score(k, n, m, cross_scores, valid, remaining)
            if mode == "cross":
                return (reuse, int(cross_scores[n]), soft, int(n))
            if mode == "risk":
                return (
                    reuse,
                    float(Q[k, n, m]),
                    float(T_metric[k, n, m]),
                    soft,
                    int(n),
                )
            if mode == "gnn":
                return (
                    reuse,
                    -float(assign_intensity[k, n, m]),
                    soft,
                    int(n),
                )
            return (reuse, soft, int(n))

        for n in sorted(valid_list, key=key):
            if remaining <= tol:
                break
            p_hit = 1.0 - float(Q[k, n, m])
            shots = int(np.ceil(max(remaining - tol, 0.0) / p_hit))
            shots = min(shots, int(inventory[k, n]))
            if shots <= 0:
                continue
            matrix[k, n, m] += shots
            inventory[k, n] -= shots
            remaining -= shots * p_hit

        return remaining <= tol

    def allocate_target(m, scheme, matrix, inventory, mode="balanced"):
        # Routes already assigned to the same target do not conflict with each other,
        # so this score is constant while allocating all weapon types for target m.
        cross_scores = target_cross_scores(matrix, m)
        weapons = sorted(
            [k for k in range(K) if U[scheme, k, m] > tol],
            key=lambda k: weapon_priority(k, m, scheme, inventory),
        )
        for k in weapons:
            delivered = float(
                np.sum((1.0 - Q[k, :, m]) * matrix[k, :, m])
            )
            remaining_need = max(float(U[scheme, k, m]) - delivered, 0.0)
            if not greedy_allocate(
                remaining_need, k, m, inventory, matrix, cross_scores, mode=mode
            ):
                return False
        return True

    def clear_target(m, matrix, inventory):
        released = matrix[:, :, m].copy()
        inventory += released
        matrix[:, :, m] = 0

    def remaining_from_matrix(matrix):
        used = np.sum(matrix, axis=2).astype(np.int64)
        return inventory_cap - used

    def try_target_schemes(m, matrix, inventory, choices, mode="balanced"):
        old = int(choices[m])
        for scheme in scheme_orders[m]:
            candidate_A = matrix.copy()
            candidate_B = inventory.copy()
            if allocate_target(m, scheme, candidate_A, candidate_B, mode=mode):
                matrix[:] = candidate_A
                inventory[:] = candidate_B
                choices[m] = int(scheme)
                return True
        choices[m] = old if 0 <= old < L else preferred_scheme(m)
        return False

    seed_preserving_target_resets = 0

    def construct_solution(order):
        nonlocal seed_preserving_target_resets
        if seed_template is None:
            matrix = np.zeros((K, N, M), dtype=np.int64)
            inventory = inventory_cap.copy()
            choices = raw_scheme_idx.copy()
        else:
            matrix = seed_template.copy()
            inventory = remaining_from_matrix(matrix)
            choices = seed_choices.copy()
        resets = 0
        for m in order:
            if seed_template is not None:
                candidate_A = matrix.copy()
                candidate_B = inventory.copy()
                if allocate_target(
                    m, int(choices[m]), candidate_A, candidate_B, mode="balanced"
                ):
                    matrix[:], inventory[:] = candidate_A, candidate_B
                    continue
                # Only the failed target is reset; allocations belonging to
                # every other target remain inherited from the neural seed.
                clear_target(m, matrix, inventory)
                resets += 1
            if not try_target_schemes(m, matrix, inventory, choices, mode="balanced"):
                return None
        seed_preserving_target_resets = int(resets)
        return matrix, inventory, choices

    construction_start = time.perf_counter()
    construction_attempts = 1
    construction_order_name = "scarcity"
    initial_state = construct_solution(scarcity_order)
    if initial_state is None:
        recovery_orders = [
            ("reverse_scarcity", list(reversed(scarcity_order))),
            ("natural", list(range(M))),
            ("target_y", np.argsort(target_coords[:, 1]).tolist()),
        ]
        for order_name, order in recovery_orders:
            construction_attempts += 1
            initial_state = construct_solution(order)
            if initial_state is not None:
                construction_order_name = order_name
                break

    # The original four-order greedy constructor can consume inventory needed
    # by a later target and used to return an all-zero, hard-infeasible plan.
    # Try additional deterministic target orders as a final Standard-only
    # rescue. This reads only the physical tensors and never uses stored
    # reference decisions or another optimizer.
    physical_rescue_used = False
    physical_rescue_method = ""
    if initial_state is None:
        rescue_rng = np.random.default_rng(2026)
        for rescue_index in range(64):
            construction_attempts += 1
            rescue_order = rescue_rng.permutation(M).tolist()
            initial_state = construct_solution(rescue_order)
            if initial_state is not None:
                construction_order_name = "deterministic_permutation_rescue"
                physical_rescue_used = True
                physical_rescue_method = f"permutation_{rescue_index + 1}"
                break
    if initial_state is None:
        raise RuntimeError(
            "DeepWTA physical feasibility construction failed after fixed and "
            "deterministic permutation rescue attempts"
        )
    construction_runtime_ms = 1000.0 * (time.perf_counter() - construction_start)

    initial_ok = initial_state is not None
    if initial_state is not None:
        A[:], remain_B[:], scheme_idx[:] = initial_state
    else:
        scheme_idx[:] = np.asarray([preferred_scheme(m) for m in range(M)], dtype=np.int64)

    initial_scheme_idx = scheme_idx.copy()

    def full_cross_pairs(matrix):
        return crossing_pairs_from_allocations(
            matrix, base_coords, target_coords, tol=tol
        )

    def replacement_cross_count(candidate_matrix, affected_targets, current_pairs):
        affected = set(int(m) for m in affected_targets)
        fixed = 0
        for route_a, route_b in current_pairs:
            if route_a[1] not in affected and route_b[1] not in affected:
                fixed += 1

        all_routes = active_routes(candidate_matrix, tol=tol)
        external = [r for r in all_routes if r[1] not in affected]
        changed = [r for r in all_routes if r[1] in affected]

        count = fixed
        for route_a in changed:
            for route_b in external:
                if cached_route_conflict(route_a, route_b):
                    count += 1
        for i in range(len(changed) - 1):
            for j in range(i + 1, len(changed)):
                if cached_route_conflict(changed[i], changed[j]):
                    count += 1
        return int(count)

    def selected_demand_total(choices):
        S, _ = selected_plan(U, G, choices, tol=tol)
        return max(float(np.sum(np.maximum(S, 0.0))), 1e-6)

    def soft_metrics(matrix, choices, cross_override=None):
        positive = np.maximum(matrix, 0.0)
        ammo = float(np.sum(positive))
        safe_ammo = max(ammo, 1e-12)
        if cross_override is None:
            cross_value = len(full_cross_pairs(matrix))
        else:
            cross_value = int(cross_override)
        avg_flight = (
            float(np.sum(positive * T_metric) / safe_ammo) if ammo > tol else 0.0
        )
        avg_inter = (
            float(np.sum(positive * Q) / safe_ammo) if ammo > tol else 0.0
        )
        ammo_norm = ammo / selected_demand_total(choices)
        cross_norm = float(cross_value) / cross_base

        route_list = active_routes(matrix, tol=tol)
        route_count = len(route_list)
        total_route_pairs = route_count * (route_count - 1) // 2
        if route_count > 0:
            route_array = np.asarray(route_list, dtype=np.int64)
            per_base = np.bincount(route_array[:, 0], minlength=N)
            per_target = np.bincount(route_array[:, 1], minlength=M)
            shared_base_pairs = int(np.sum(per_base * (per_base - 1) // 2))
            shared_target_pairs = int(np.sum(per_target * (per_target - 1) // 2))
            eligible_pairs = max(
                int(total_route_pairs - shared_base_pairs - shared_target_pairs),
                0,
            )
        else:
            eligible_pairs = 0
        crossing_pair_rate = (
            float(cross_value) / eligible_pairs
            if eligible_pairs > 0
            else 0.0
        )

        cost = (
            _W_AMMO * ammo_norm
            + _W_CROSS * cross_norm
            + _W_FLIGHT * avg_flight
            + _W_INTERCEPT * avg_inter
        )
        return {
            "cost": float(cost),
            "ammo": ammo,
            "ammo_norm": float(ammo_norm),
            "cross": int(cross_value),
            "cross_norm": float(cross_norm),
            "eligible_active_route_pairs": int(eligible_pairs),
            "crossing_pair_rate": float(crossing_pair_rate),
            "active_route_count": int(route_count),
            "flight": avg_flight,
            "intercept": avg_inter,
        }

    def hard_report(matrix, choices):
        return validate_solution(
            matrix,
            choices,
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

    def target_met(matrix, choices, m):
        S, _ = selected_plan(U, G, choices, tol=tol)
        effective = np.sum(matrix[:, :, m] * (1.0 - Q[:, :, m]), axis=1)
        return bool(np.all(effective + tol >= S[:, m]))

    def apply_state(matrix, choices):
        A[:] = matrix
        scheme_idx[:] = choices
        remain_B[:] = remaining_from_matrix(matrix)

    initial_report = hard_report(A, scheme_idx)
    initial_pairs = full_cross_pairs(A)
    initial_soft = soft_metrics(A, scheme_idx, len(initial_pairs))

    if max_passes is None:
        max_passes = min(20, max(8, M // 8 + 2))
    else:
        max_passes = max(0, int(max_passes))

    repair_passes = 0
    repair_moves = 0
    allocation_moves = 0
    scheme_moves = 0
    soft_stalled = False

    def try_same_scheme_rebuild(m, current_pairs, current_metrics):
        current_scheme = int(scheme_idx[m])
        base_A = A.copy()
        base_B = remain_B.copy()
        clear_target(m, base_A, base_B)

        best = None
        best_cost = current_metrics["cost"]
        for mode in ("cross", "risk", "gnn"):
            candidate_A = base_A.copy()
            candidate_B = base_B.copy()
            if not allocate_target(
                m, current_scheme, candidate_A, candidate_B, mode=mode
            ):
                continue
            candidate_choice = scheme_idx.copy()
            new_cross = replacement_cross_count(candidate_A, [m], current_pairs)
            metrics = soft_metrics(candidate_A, candidate_choice, new_cross)
            if metrics["cost"] < best_cost - _MIN_SOFT_GAIN:
                best_cost = metrics["cost"]
                best = (candidate_A, candidate_choice, metrics)

        if best is None:
            return False, current_metrics
        apply_state(best[0], best[1])
        return True, best[2]

    def try_two_opt(route_a, route_b, current_pairs, current_metrics):
        n1, m1 = route_a
        n2, m2 = route_b
        common_k = np.flatnonzero((A[:, n1, m1] > 0) & (A[:, n2, m2] > 0))
        if common_k.size == 0:
            return False, current_metrics

        best = None
        best_cost = current_metrics["cost"]
        for k in common_k:
            if X[k, n1, m2] <= 0.5 or X[k, n2, m1] <= 0.5:
                continue
            max_swap = int(min(A[k, n1, m1], A[k, n2, m2]))
            if max_swap <= 0:
                continue

            for swap in sorted(set([1, max_swap])):
                candidate = A.copy()
                candidate[k, n1, m1] -= swap
                candidate[k, n2, m2] -= swap
                candidate[k, n1, m2] += swap
                candidate[k, n2, m1] += swap

                if not target_met(candidate, scheme_idx, m1):
                    continue
                if not target_met(candidate, scheme_idx, m2):
                    continue

                new_cross = replacement_cross_count(
                    candidate, [m1, m2], current_pairs
                )
                metrics = soft_metrics(candidate, scheme_idx, new_cross)
                if metrics["cost"] < best_cost - _MIN_SOFT_GAIN:
                    best_cost = metrics["cost"]
                    best = (candidate, metrics)

        if best is None:
            return False, current_metrics
        apply_state(best[0], scheme_idx.copy())
        return True, best[1]

    def alternative_schemes(m):
        current = int(scheme_idx[m])
        return [s for s in scheme_orders[m] if s != current][:_LOCAL_SCHEME_TOPK]

    def try_single_scheme_improvement(m, current_pairs, current_metrics):
        alternatives = alternative_schemes(m)
        if not alternatives:
            return False, current_metrics

        base_A = A.copy()
        base_B = remain_B.copy()
        clear_target(m, base_A, base_B)
        best = None
        best_cost = current_metrics["cost"]

        for scheme in alternatives:
            for mode in ("balanced", "cross"):
                candidate_A = base_A.copy()
                candidate_B = base_B.copy()
                if not allocate_target(m, scheme, candidate_A, candidate_B, mode=mode):
                    continue
                candidate_choice = scheme_idx.copy()
                candidate_choice[m] = int(scheme)
                new_cross = replacement_cross_count(candidate_A, [m], current_pairs)
                metrics = soft_metrics(candidate_A, candidate_choice, new_cross)
                if metrics["cost"] < best_cost - _MIN_SOFT_GAIN:
                    best_cost = metrics["cost"]
                    best = (candidate_A, candidate_choice, metrics)

        if best is None:
            return False, current_metrics
        apply_state(best[0], best[1])
        return True, best[2]

    def local_scheme_choices(m):
        current = int(scheme_idx[m])
        return [current] + alternative_schemes(m)[:2]

    def try_pair_scheme_improvement(m1, m2, current_pairs, current_metrics):
        if m1 == m2:
            return False, current_metrics

        base_A = A.copy()
        base_B = remain_B.copy()
        clear_target(m1, base_A, base_B)
        clear_target(m2, base_A, base_B)
        best = None
        best_cost = current_metrics["cost"]

        for s1, s2 in product(local_scheme_choices(m1), local_scheme_choices(m2)):
            for order in ((m1, m2), (m2, m1)):
                candidate_A = base_A.copy()
                candidate_B = base_B.copy()
                candidate_choice = scheme_idx.copy()
                chosen = {m1: int(s1), m2: int(s2)}
                ok = True
                for m in order:
                    if not allocate_target(
                        m, chosen[m], candidate_A, candidate_B, mode="balanced"
                    ):
                        ok = False
                        break
                    candidate_choice[m] = chosen[m]
                if not ok:
                    continue

                new_cross = replacement_cross_count(
                    candidate_A, [m1, m2], current_pairs
                )
                metrics = soft_metrics(candidate_A, candidate_choice, new_cross)
                if metrics["cost"] < best_cost - _MIN_SOFT_GAIN:
                    best_cost = metrics["cost"]
                    best = (candidate_A, candidate_choice, metrics)

        if best is None:
            return False, current_metrics
        apply_state(best[0], best[1])
        return True, best[2]

    current_metrics = initial_soft.copy()

    local_improvement_start = time.perf_counter()
    if initial_ok and bool(initial_report.get("hard_feasible", False)):
        for pass_id in range(max_passes):
            current_pairs = full_cross_pairs(A)
            if not current_pairs:
                break

            repair_passes = pass_id + 1
            degree = np.zeros(M, dtype=np.int64)
            for route_a, route_b in current_pairs:
                degree[route_a[1]] += 1
                degree[route_b[1]] += 1
            conflict_targets = np.flatnonzero(degree > 0).tolist()
            conflict_targets.sort(key=lambda m: (-int(degree[m]), int(m)))

            changed = False

            for m in conflict_targets[:_LOCAL_TARGET_LIMIT]:
                changed, current_metrics = try_same_scheme_rebuild(
                    m, current_pairs, current_metrics
                )
                if changed:
                    allocation_moves += 1
                    break

            if not changed:
                for route_a, route_b in current_pairs[:_LOCAL_PAIR_LIMIT]:
                    changed, current_metrics = try_two_opt(
                        route_a, route_b, current_pairs, current_metrics
                    )
                    if changed:
                        allocation_moves += 1
                        break

            if not changed:
                for m in conflict_targets[:_LOCAL_TARGET_LIMIT]:
                    changed, current_metrics = try_single_scheme_improvement(
                        m, current_pairs, current_metrics
                    )
                    if changed:
                        scheme_moves += 1
                        break

            if not changed:
                seen = set()
                for route_a, route_b in current_pairs[:_LOCAL_PAIR_LIMIT]:
                    m1, m2 = int(route_a[1]), int(route_b[1])
                    key = tuple(sorted((m1, m2)))
                    if m1 == m2 or key in seen:
                        continue
                    seen.add(key)
                    changed, current_metrics = try_pair_scheme_improvement(
                        m1, m2, current_pairs, current_metrics
                    )
                    if changed:
                        scheme_moves += 1
                        break

            if not changed:
                soft_stalled = True
                break

            repair_moves += 1

    local_improvement_runtime_ms = 1000.0 * (
        time.perf_counter() - local_improvement_start
    )

    validation_start = time.perf_counter()
    final_report = hard_report(A, scheme_idx)
    final_pairs = full_cross_pairs(A)
    final_soft = soft_metrics(A, scheme_idx, len(final_pairs))
    validation_runtime_ms = 1000.0 * (
        time.perf_counter() - validation_start
    )
    final_scheme_idx = scheme_idx.copy()

    raw_level_ok = G[raw_scheme_idx, np.arange(M)] >= G_th - tol
    raw_to_initial = initial_scheme_idx != raw_scheme_idx
    raw_to_final = final_scheme_idx != raw_scheme_idx
    initial_to_final = final_scheme_idx != initial_scheme_idx

    final_report["solver_component"] = CONSTRUCTION_ALGORITHM
    final_report["crossing_is_soft_objective"] = True
    final_report["construction_runtime_ms"] = float(construction_runtime_ms)
    final_report["local_improvement_runtime_ms"] = float(local_improvement_runtime_ms)
    final_report["validation_runtime_ms"] = float(validation_runtime_ms)
    final_report["repair_total_runtime_ms_internal"] = float(
        1000.0 * (time.perf_counter() - repair_total_start)
    )
    final_report["construction_attempts"] = int(construction_attempts)
    final_report["construction_order_name"] = str(construction_order_name)
    final_report["physical_feasibility_rescue_used"] = bool(physical_rescue_used)
    final_report["physical_feasibility_rescue_method"] = str(physical_rescue_method)
    final_report["initial_hard_feasible"] = bool(
        initial_report.get("hard_feasible", False)
    )
    final_report["initial_allocation_complete"] = bool(initial_ok)
    final_report["seed_preserving_fallback_requested"] = bool(
        seed_preserving_requested
    )
    final_report["seed_preserving_fallback_succeeded"] = bool(
        seed_preserving_requested and initial_ok and not physical_rescue_used
    )
    final_report["seed_preserving_trimmed_shots"] = int(seed_trimmed_shots)
    final_report["seed_preserving_target_resets"] = int(
        seed_preserving_target_resets
    )
    final_report["seed_preserving_kept_seed_shots"] = float(
        np.sum(np.minimum(seed_template, A))
        if seed_template is not None
        else 0.0
    )
    final_report["all_targets_have_level_feasible_scheme"] = bool(np.all(scheme_has_level))
    final_report["target_processing_order"] = [int(m) for m in scarcity_order]
    final_report["crossing_conflict_count_before_repair"] = int(len(initial_pairs))
    final_report["crossing_conflict_count_after_repair"] = int(len(final_pairs))
    final_report["repair_passes"] = int(repair_passes)
    final_report["repair_moves"] = int(repair_moves)
    final_report["allocation_improvement_moves"] = int(allocation_moves)
    final_report["scheme_improvement_moves"] = int(scheme_moves)
    final_report["repair_stalled"] = bool(soft_stalled)
    final_report["max_repair_passes"] = int(max_passes)
    final_report["soft_cost_before_repair"] = float(initial_soft["cost"])
    final_report["soft_cost_after_repair"] = float(final_soft["cost"])
    final_report["soft_cost_improvement"] = float(initial_soft["cost"] - final_soft["cost"])
    final_report["soft_components_before"] = initial_soft
    final_report["soft_components_after"] = final_soft

    final_report["raw_scheme_idx"] = raw_scheme_idx.tolist()
    final_report["initial_scheme_idx"] = initial_scheme_idx.tolist()
    final_report["final_scheme_idx"] = final_scheme_idx.tolist()
    final_report["raw_scheme_level_feasible_count"] = int(np.sum(raw_level_ok))
    final_report["raw_scheme_level_feasible_rate"] = float(np.mean(raw_level_ok))
    final_report["initial_scheme_change_count"] = int(np.sum(raw_to_initial))
    final_report["final_scheme_change_count"] = int(np.sum(raw_to_final))
    final_report["scheme_changes_due_to_level_count"] = int(
        np.sum(raw_to_initial & (~raw_level_ok))
    )
    final_report["scheme_changes_due_to_construction_count"] = int(
        np.sum(raw_to_initial & raw_level_ok)
    )
    final_report["scheme_changes_during_repair_count"] = int(np.sum(initial_to_final))
    final_report["mean_raw_scheme_prob"] = float(
        np.mean(scheme_probs[np.arange(M), raw_scheme_idx])
    )
    final_report["mean_initial_scheme_prob"] = float(
        np.mean(scheme_probs[np.arange(M), initial_scheme_idx])
    )
    final_report["mean_final_scheme_prob"] = float(
        np.mean(scheme_probs[np.arange(M), final_scheme_idx])
    )
    final_report["scheme_selection_policy"] = (
        "GNN-first level-feasible scheme; fallback only when construction fails; "
        "later scheme changes require a strict reduction in the weighted soft objective"
    )
    final_report["soft_objective_policy"] = (
        "10*ammo_norm + 80*geo_cross_norm + 30*avg_flight + 10*avg_interception"
    )

    if return_report:
        return A, scheme_idx, final_report
    return A, scheme_idx
