from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np

from configs.deepwta_solver_config import DeepWTASolverConfig
from deepwta.mechanisms.conflict_index import LazyConflictIndex
from deepwta.mechanisms.incremental_state import (
    IncrementalRepairState,
    TacticalWeights,
)
from deepwta.mechanisms.integer_decoder import (
    build_target_route_candidates,
    residual_aware_integerize,
    residual_route_mass,
    soft_route_mass,
)
from deepwta.utils.geometry import route_conflict
from deepwta.utils.validator import selected_plan, validate_solution


STANDARD_ALGORITHM = "deepwta-standard"
Route = tuple[int, int]


def _hard_level_feasible_schemes(
    scheme_probs: np.ndarray,
    G: np.ndarray,
    G_th: np.ndarray,
    tol: float,
) -> np.ndarray:
    M, _ = scheme_probs.shape
    selected = np.zeros(M, dtype=np.int64)
    for m in range(M):
        valid = np.flatnonzero(G[:, m] >= G_th[m] - tol)
        if valid.size == 0:
            selected[m] = int(np.argmax(G[:, m]))
            continue
        selected[m] = int(valid[np.argmax(scheme_probs[m, valid])])
    return selected


def _distance_metric(
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    flight_times: np.ndarray | None,
    K: int,
    N: int,
    M: int,
) -> tuple[np.ndarray, np.ndarray]:
    route_distance = np.linalg.norm(
        base_coords[:, None, :] - target_coords[None, :, :],
        axis=2,
    )
    max_distance = max(float(np.max(route_distance)), 1e-12)
    distance_norm = route_distance / max_distance

    if flight_times is None:
        return distance_norm, np.broadcast_to(
            distance_norm[None, :, :],
            (K, N, M),
        ).copy()

    T = np.asarray(flight_times, dtype=np.float64)
    if T.shape == (N, M):
        T_metric = np.broadcast_to(T[None, :, :], (K, N, M)).copy()
        scalar = T
    elif T.shape == (1, N, M):
        T_metric = np.broadcast_to(T, (K, N, M)).copy()
        scalar = T[0]
    elif T.shape == (K, N, M):
        T_metric = T.copy()
        scalar = np.mean(T, axis=0)
    else:
        raise ValueError(
            "flight_times must have shape [N,M], [1,N,M], or [K,N,M]"
        )
    if not np.all(np.isfinite(T_metric)):
        raise ValueError("flight_times must be finite")
    return np.asarray(scalar, dtype=np.float64), T_metric


def _direct_delta_cross(
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    route: Route,
    active_routes: set[Route],
) -> int:
    if route in active_routes:
        return 0
    return int(
        sum(
            1
            for other in active_routes
            if route_conflict(
                base_coords,
                target_coords,
                route,
                other,
            )
        )
    )


def _target_candidate_bases(
    state: IncrementalRepairState,
    candidates,
    target: int,
) -> list[int]:
    """Current Top-K plus already-active floor/repair routes for this target."""
    m = int(target)
    ordered: list[int] = []
    seen: set[int] = set()

    # Existing routes are always eligible even if outside initial Top-K.
    active_bases = [i for i, t in state.active_routes if t == m]
    active_bases.sort(key=lambda i: (-int(state.route_load[i, m]), int(i)))
    for i in active_bases:
        if i not in seen:
            ordered.append(int(i))
            seen.add(int(i))

    for i in candidates[m].current().tolist():
        i = int(i)
        if i not in seen:
            ordered.append(i)
            seen.add(i)
    return ordered


def _target_capacity_ratio(
    state: IncrementalRepairState,
    candidates,
    target: int,
) -> tuple[float, float, int]:
    """Scarcity-first priority using all fallback bases, not only current Top-K."""
    m = int(target)
    ratios = []
    for k in range(state.K):
        shortage = float(state.shortage[k, m])
        if shortage <= state.tol:
            continue
        capacity = 0.0
        for i in candidates[m].all().tolist():
            i = int(i)
            if state.remaining_inventory[k, i] <= 0:
                continue
            if state.X[k, i, m] <= 0.5:
                continue
            survival = max(1.0 - float(state.Q[k, i, m]), 0.0)
            capacity += float(state.remaining_inventory[k, i]) * survival
        ratios.append(capacity / max(shortage, state.tol))

    if not ratios:
        return (float("inf"), 0.0, m)
    return (float(min(ratios)), float(np.sum(state.shortage[:, m])), m)


def _route_progress_features(
    state: IncrementalRepairState,
    base: int,
    target: int,
) -> tuple[int, float, float, float] | None:
    """Return coverage features for one route against current target shortage.

    Returns:
      cover_count,
      coverage_fraction (0..1-ish across weapon shortages),
      effective_capacity_used_for_shortage,
      shortage-weighted avg interception.
    """
    i = int(base)
    m = int(target)
    total_shortage = float(np.sum(state.shortage[:, m]))
    if total_shortage <= state.tol:
        return None

    cover_count = 0
    covered = 0.0
    q_weighted = 0.0
    q_weight = 0.0

    for k in range(state.K):
        shortage = float(state.shortage[k, m])
        if shortage <= state.tol:
            continue
        if state.remaining_inventory[k, i] <= 0:
            continue
        if state.X[k, i, m] <= 0.5:
            continue
        survival = max(1.0 - float(state.Q[k, i, m]), 0.0)
        if survival <= state.tol:
            continue

        effective_cap = float(state.remaining_inventory[k, i]) * survival
        contribution = min(shortage, effective_cap)
        if contribution <= state.tol:
            continue

        cover_count += 1
        covered += contribution
        q_weighted += shortage * float(state.Q[k, i, m])
        q_weight += shortage

    if cover_count == 0:
        return None

    coverage_fraction = covered / max(total_shortage, state.tol)
    avg_q = q_weighted / max(q_weight, state.tol)
    return (
        int(cover_count),
        float(coverage_fraction),
        float(covered),
        float(avg_q),
    )


def _select_route_for_target(
    state: IncrementalRepairState,
    candidates,
    target: int,
    route_mass: np.ndarray,
    residual_mass: np.ndarray,
    D_norm: np.ndarray,
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    cfg: DeepWTASolverConfig,
) -> tuple[int | None, dict]:
    """Lexicographic target-level route selection.

    No scalar alpha tuning is used for crossing. This avoids mixing exact
    integer crossing counts with [0,1] GNN/residual quantities.
    """
    m = int(target)
    bases = _target_candidate_bases(state, candidates, m)
    if not bases:
        return None, {}

    max_gnn = max(float(np.max(route_mass[:, m])), 1e-12)
    max_res = max(float(np.max(residual_mass[:, m])), 1e-12)

    scored = []
    detail_by_base = {}
    for i in bases:
        features = _route_progress_features(state, i, m)
        if features is None:
            continue
        cover_count, coverage_fraction, covered, avg_q = features
        route = (int(i), m)
        active = route in state.active_routes

        if active:
            delta_cross = 0
        elif True:
            delta_cross = state.conflict_index.query_delta_open(
                route,
                state.active_routes,
            )
        else:
            delta_cross = _direct_delta_cross(
                base_coords,
                target_coords,
                route,
                state.active_routes,
            )

        gnn_norm = float(route_mass[i, m] / max_gnn)
        residual_norm = float(residual_mass[i, m] / max_res)
        active_rank = 0 if (cfg.reuse_active_routes_first and active) else 1

        # Lower tuple is better. Crossing is exact and lexicographic; learned
        # scores only break ties after topology/capacity considerations.
        key = (
            int(active_rank),
            int(delta_cross),
            -float(coverage_fraction),
            -int(cover_count),
            -float(covered),
            -float(gnn_norm),
            -float(residual_norm),
            float(D_norm[i, m]),
            float(avg_q),
            int(i),
        )
        scored.append((key, int(i)))
        detail_by_base[int(i)] = {
            "active": bool(active),
            "delta_cross": int(delta_cross),
            "coverage_fraction": float(coverage_fraction),
            "cover_count": int(cover_count),
            "covered": float(covered),
            "gnn_norm": float(gnn_norm),
            "residual_norm": float(residual_norm),
        }

    if not scored:
        return None, {}
    scored.sort(key=lambda item: item[0])
    best_i = int(scored[0][1])
    return best_i, detail_by_base[best_i]


def _bulk_fill_selected_route(
    state: IncrementalRepairState,
    base: int,
    target: int,
) -> tuple[int, int]:
    """Fill as much current shortage as possible on one selected route.

    Returns (shots_added, weapon_bulk_operations).
    """
    i = int(base)
    m = int(target)
    shots_added = 0
    operations = 0

    # Hardest/least-surviving weapon shortage first on the chosen route. Since
    # inventories are weapon-specific this ordering is mostly deterministic
    # tie-breaking, but it makes logs reproducible and explicit.
    weapons = [k for k in range(state.K) if state.shortage[k, m] > state.tol]
    weapons.sort(
        key=lambda k: (
            max(1.0 - float(state.Q[k, i, m]), 0.0),
            -float(state.shortage[k, m]),
            int(k),
        )
    )

    for k in weapons:
        shortage = float(state.shortage[k, m])
        if shortage <= state.tol:
            continue
        if state.remaining_inventory[k, i] <= 0 or state.X[k, i, m] <= 0.5:
            continue
        survival = max(1.0 - float(state.Q[k, i, m]), 0.0)
        if survival <= state.tol:
            continue

        required = int(math.ceil(max(shortage - state.tol, 0.0) / survival))
        shots = min(required, int(state.remaining_inventory[k, i]))
        if shots <= 0:
            continue
        state.add_shots(k, i, m, shots)
        shots_added += shots
        operations += 1

    return int(shots_added), int(operations)


def _fill_shortages_route_concentrated(
    state: IncrementalRepairState,
    assign_intensity: np.ndarray,
    residual: np.ndarray,
    candidates,
    D_norm: np.ndarray,
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    cfg: DeepWTASolverConfig,
) -> dict:
    route_mass = soft_route_mass(assign_intensity)
    residual_mass = residual_route_mass(residual)

    expansions = 0
    shots_added = 0
    bulk_operations = 0
    routes_opened = 0
    route_selections = 0
    active_reuse_selections = 0
    failed_target: int | None = None

    while True:
        pending_targets = [
            m
            for m in range(state.M)
            if float(np.max(state.shortage[:, m])) > cfg.tol
        ]
        if not pending_targets:
            break

        pending_targets.sort(
            key=lambda m: _target_capacity_ratio(state, candidates, m)
        )
        m = int(pending_targets[0])

        while float(np.max(state.shortage[:, m])) > cfg.tol:
            best_i, detail = _select_route_for_target(
                state,
                candidates,
                m,
                route_mass,
                residual_mass,
                D_norm,
                base_coords,
                target_coords,
                cfg,
            )

            if best_i is None:
                if candidates[m].expand():
                    expansions += 1
                    continue
                failed_target = m
                return {
                    "success": False,
                    "failed_target": int(failed_target),
                    "candidate_expansions": int(expansions),
                    "shots_added": int(shots_added),
                    "bulk_operations": int(bulk_operations),
                    "routes_opened": int(routes_opened),
                    "route_selections": int(route_selections),
                    "active_reuse_selections": int(active_reuse_selections),
                }

            route = (int(best_i), m)
            was_active = route in state.active_routes
            added, operations = _bulk_fill_selected_route(
                state,
                best_i,
                m,
            )
            route_selections += 1
            if was_active:
                active_reuse_selections += 1
            elif added > 0:
                routes_opened += 1
            shots_added += added
            bulk_operations += operations

            if added <= 0:
                # The selected current candidates cannot make progress; expand
                # before declaring failure. This keeps Top-K an accelerator,
                # never a hard feasibility cut.
                if candidates[m].expand():
                    expansions += 1
                    continue
                failed_target = m
                return {
                    "success": False,
                    "failed_target": int(failed_target),
                    "candidate_expansions": int(expansions),
                    "shots_added": int(shots_added),
                    "bulk_operations": int(bulk_operations),
                    "routes_opened": int(routes_opened),
                    "route_selections": int(route_selections),
                    "active_reuse_selections": int(active_reuse_selections),
                }

    return {
        "success": True,
        "failed_target": None,
        "candidate_expansions": int(expansions),
        "shots_added": int(shots_added),
        "bulk_operations": int(bulk_operations),
        "routes_opened": int(routes_opened),
        "route_selections": int(route_selections),
        "active_reuse_selections": int(active_reuse_selections),
    }


def _all_target_destinations(state, candidates, target: int, exclude: int) -> list[int]:
    m = int(target)
    exclude = int(exclude)
    active = [i for i, t in state.active_routes if t == m and i != exclude]
    active.sort(key=lambda i: (-int(state.route_load[i, m]), int(i)))

    result: list[int] = []
    seen = {exclude}
    for i in active + candidates[m].current().tolist() + candidates[m].all().tolist():
        i = int(i)
        if i in seen:
            continue
        result.append(i)
        seen.add(i)
    return result


def _route_consolidation_pass(
    state: IncrementalRepairState,
    candidates,
    hotspot_targets: Iterable[int],
    cfg: DeepWTASolverConfig,
    deadline: float | None,
) -> int:
    moves = 0

    for m in hotspot_targets:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        m = int(m)
        active_bases = [i for i, t in state.active_routes if t == m]
        if len(active_bases) <= 1:
            continue

        # Close high-conflict, low-load routes first.
        active_bases.sort(
            key=lambda i: (
                -state.conflict_index.route_crossing_degree(
                    (int(i), m),
                    state.active_routes,
                ),
                int(state.route_load[i, m]),
                int(i),
            )
        )

        trials = 0
        for old_i in active_bases:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if (int(old_i), m) not in state.active_routes:
                continue

            best: tuple[float, int] | None = None
            destinations = _all_target_destinations(
                state,
                candidates,
                m,
                int(old_i),
            )
            # Existing destination routes first; consolidation should avoid
            # replacing one route with another unless the exact cost warrants it.
            destinations.sort(
                key=lambda i: (
                    0 if (int(i), m) in state.active_routes else 1,
                    -int(state.route_load[i, m]),
                    int(i),
                )
            )

            for new_i in destinations:
                if trials >= cfg.max_route_consolidation_trials_per_target:
                    break
                trials += 1
                delta = state.route_transfer_delta(
                    int(old_i),
                    int(new_i),
                    m,
                )
                if delta is None:
                    continue
                if best is None or delta < best[0]:
                    best = (float(delta), int(new_i))

            if best is not None and best[0] < -1e-10:
                state.apply_route_transfer(
                    int(old_i),
                    best[1],
                    m,
                )
                moves += 1

    return int(moves)


def _one_shot_transfer_pass(
    state: IncrementalRepairState,
    candidates,
    hotspot_targets: Iterable[int],
    cfg: DeepWTASolverConfig,
    deadline: float | None,
) -> int:
    moves = 0

    for m in hotspot_targets:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        m = int(m)
        trials = 0
        best_move = None

        for old_i, target in sorted(state.active_routes):
            if target != m:
                continue
            for k in np.flatnonzero(state.A[:, old_i, m] > 0).tolist():
                for new_i in _all_target_destinations(
                    state,
                    candidates,
                    m,
                    int(old_i),
                ):
                    if trials >= cfg.max_transfer_trials_per_target:
                        break
                    trials += 1

                    # Avoid opening a new route unless this shot would close the
                    # old route; otherwise crossing topology cannot improve.
                    old_closes = state.route_load[old_i, m] == 1
                    new_active = (int(new_i), m) in state.active_routes
                    if not old_closes and not new_active:
                        continue

                    delta = state.bulk_transfer_delta(
                        int(k),
                        int(old_i),
                        int(new_i),
                        m,
                        1,
                    )
                    if delta is None:
                        continue
                    if best_move is None or delta < best_move[0]:
                        best_move = (
                            float(delta),
                            int(k),
                            int(old_i),
                            int(new_i),
                            m,
                        )
                if trials >= cfg.max_transfer_trials_per_target:
                    break
            if trials >= cfg.max_transfer_trials_per_target:
                break

        if best_move is not None and best_move[0] < -1e-10:
            _, k, old_i, new_i, m = best_move
            state.apply_bulk_transfer(k, old_i, new_i, m, 1)
            moves += 1

    return int(moves)


def _rare_scheme_change_pass(
    state: IncrementalRepairState,
    scheme_probs: np.ndarray,
    G: np.ndarray,
    G_th: np.ndarray,
    assign_intensity: np.ndarray,
    hotspot_targets: Iterable[int],
    cfg: DeepWTASolverConfig,
    deadline: float | None,
) -> int:
    if not cfg.enable_rare_scheme_change:
        return 0

    moves = 0
    for m in list(hotspot_targets)[: cfg.max_scheme_targets]:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        m = int(m)
        valid = np.flatnonzero(G[:, m] >= G_th[m] - cfg.tol)
        if valid.size <= 1:
            continue
        ordered = valid[np.argsort(-scheme_probs[m, valid], kind="stable")]
        top = ordered[:2]
        current = int(state.scheme_idx[m])
        if len(top) < 2 or current not in top:
            continue
        alternative = int(top[0] if int(top[1]) == current else top[1])
        gap = abs(
            float(scheme_probs[m, current])
            - float(scheme_probs[m, alternative])
        )
        if gap > cfg.scheme_probability_gap:
            continue

        candidate = state.clone()
        if not candidate.set_scheme_if_already_feasible(m, alternative):
            continue

        # Remove redundant low-guidance shots without rebuilding the target.
        entries = []
        for k in range(candidate.K):
            for i in np.flatnonzero(candidate.A[k, :, m] > 0).tolist():
                entries.append((float(assign_intensity[k, i, m]), int(k), int(i)))
        entries.sort()

        changed = True
        while changed:
            changed = False
            for _, k, i in entries:
                if candidate.A[k, i, m] <= 0:
                    continue
                if not candidate.can_remove(k, i, m, 1):
                    continue
                before = candidate.tactical_cost()
                trial = candidate.clone()
                trial.remove_shot(k, i, m)
                if trial.tactical_cost() < before - 1e-10:
                    candidate = trial
                    changed = True

        if (
            candidate.hard_feasible_internal()
            and candidate.tactical_cost() < state.tactical_cost() - 1e-10
        ):
            state.replace_from(candidate)
            moves += 1

    return int(moves)


def _local_search(
    state: IncrementalRepairState,
    candidates,
    scheme_probs: np.ndarray,
    G: np.ndarray,
    G_th: np.ndarray,
    assign_intensity: np.ndarray,
    cfg: DeepWTASolverConfig,
    repair_start: float,
) -> dict:
    if cfg.max_passes <= 0:
        return {
            "passes": 0,
            "moves": 0,
            "route_consolidation_moves": 0,
            "transfer_moves": 0,
            "scheme_moves": 0,
            "stopped_by_budget": False,
            "stopped_by_stall": False,
        }

    budget = cfg.budget_seconds(state.N, state.M)
    deadline = None if budget is None else repair_start + budget

    passes = 0
    total_moves = 0
    consolidation_moves = 0
    transfer_moves = 0
    scheme_moves = 0
    stall_count = 0
    stopped_by_budget = False
    stopped_by_stall = False

    for pass_id in range(cfg.max_passes):
        if deadline is not None and time.perf_counter() >= deadline:
            stopped_by_budget = True
            break

        before = state.tactical_cost()
        degree = state.conflict_index.target_crossing_degree(
            state.active_routes,
            state.M,
        )
        hotspots = np.argsort(-degree)[: cfg.max_hotspot_targets]
        hotspots = [int(m) for m in hotspots.tolist() if degree[m] > 0]
        if not hotspots:
            break

        c_moves = _route_consolidation_pass(
            state,
            candidates,
            hotspots,
            cfg,
            deadline,
        )
        t_moves = _one_shot_transfer_pass(
            state,
            candidates,
            hotspots,
            cfg,
            deadline,
        )
        s_moves = _rare_scheme_change_pass(
            state,
            scheme_probs,
            G,
            G_th,
            assign_intensity,
            hotspots,
            cfg,
            deadline,
        )

        pass_moves = c_moves + t_moves + s_moves
        consolidation_moves += c_moves
        transfer_moves += t_moves
        scheme_moves += s_moves
        total_moves += pass_moves
        passes = pass_id + 1

        after = state.tactical_cost()
        relative = max(before - after, 0.0) / max(abs(before), 1e-12)
        if pass_moves == 0 or relative < cfg.min_relative_improvement:
            stall_count += 1
        else:
            stall_count = 0
        if stall_count >= cfg.stall_passes:
            stopped_by_stall = True
            break

    return {
        "passes": int(passes),
        "moves": int(total_moves),
        "route_consolidation_moves": int(consolidation_moves),
        "transfer_moves": int(transfer_moves),
        "scheme_moves": int(scheme_moves),
        "stopped_by_budget": bool(stopped_by_budget),
        "stopped_by_stall": bool(stopped_by_stall),
    }



def _size_aware_topology_polish(
    state: IncrementalRepairState,
    candidates,
    cfg: DeepWTASolverConfig,
) -> dict:
    """Deterministic topology polish for small/medium scenarios.

    Wall-clock deadlines are removed. Fixed pass, target and trial caps are the
    only stopping criteria.
    """
    started = time.perf_counter()
    bucket = cfg.size_bucket(state.N, state.M)
    before = state.tactical_components()

    enabled = (
        cfg.enable_small_medium_topology_polish
        and bucket in ("small", "medium")
        and cfg.topology_polish_passes > 0
    )
    if not enabled:
        return {
            "enabled": False,
            "size_bucket": bucket,
            "passes": 0,
            "moves": 0,
            "trials": 0,
            "stopped_by_budget": False,
            "runtime_s": float(time.perf_counter() - started),
            "crossing_before": float(before["crossing_count"]),
            "crossing_after": float(before["crossing_count"]),
            "crossing_reduction": 0.0,
            "avg_distance_before": float(before["avg_distance"]),
            "avg_distance_after": float(before["avg_distance"]),
        }

    moves = 0
    trials = 0
    passes = 0

    for pass_id in range(cfg.topology_polish_passes):
        degree = state.conflict_index.target_crossing_degree(
            state.active_routes,
            state.M,
        )
        hotspots = np.argsort(-degree, kind="stable")[: cfg.topology_polish_max_targets]
        hotspots = [int(m) for m in hotspots.tolist() if degree[m] > 0]
        if not hotspots:
            break

        pass_moves = 0
        for m in hotspots:
            active_bases = [i for i, target in state.active_routes if target == m]
            active_bases.sort(
                key=lambda i: (
                    -state.conflict_index.route_crossing_degree(
                        (int(i), m),
                        state.active_routes,
                    ),
                    -float(state.distance_metric[int(i), m]),
                    int(state.route_load[int(i), m]),
                    int(i),
                )
            )
            if not active_bases:
                continue

            destinations = _all_target_destinations(
                state,
                candidates,
                m,
                exclude=-1,
            )
            destinations.sort(
                key=lambda i: (
                    0 if (int(i), m) in state.active_routes else 1,
                    float(state.distance_metric[int(i), m]),
                    int(i),
                )
            )

            best = None
            target_trials = 0
            for old_i in active_bases:
                if (int(old_i), m) not in state.active_routes:
                    continue
                for new_i in destinations:
                    if target_trials >= cfg.topology_polish_trials_per_target:
                        break
                    new_i = int(new_i)
                    if new_i == int(old_i):
                        continue

                    target_trials += 1
                    trials += 1
                    effects = state.route_transfer_effects(
                        int(old_i),
                        new_i,
                        m,
                    )
                    if effects is None:
                        continue
                    if int(effects["delta_cross"]) > -int(
                        cfg.topology_min_crossing_reduction
                    ):
                        continue
                    if float(effects["delta_avg_distance"]) > (
                        float(cfg.topology_max_avg_distance_increase) + cfg.tol
                    ):
                        continue
                    if float(effects["delta_cost"]) >= -1e-12:
                        continue

                    key = (
                        int(effects["delta_cross"]),
                        float(effects["delta_avg_distance"]),
                        float(effects["delta_cost"]),
                        int(old_i),
                        new_i,
                    )
                    if best is None or key < best[0]:
                        best = (key, int(old_i), new_i)

                if target_trials >= cfg.topology_polish_trials_per_target:
                    break

            if best is not None:
                _, old_i, new_i = best
                state.apply_route_transfer(old_i, new_i, m)
                moves += 1
                pass_moves += 1

        passes = pass_id + 1
        if pass_moves == 0:
            break

    after = state.tactical_components()
    return {
        "enabled": True,
        "size_bucket": bucket,
        "passes": int(passes),
        "moves": int(moves),
        "trials": int(trials),
        "stopped_by_budget": False,
        "runtime_s": float(time.perf_counter() - started),
        "crossing_before": float(before["crossing_count"]),
        "crossing_after": float(after["crossing_count"]),
        "crossing_reduction": float(
            before["crossing_count"] - after["crossing_count"]
        ),
        "avg_distance_before": float(before["avg_distance"]),
        "avg_distance_after": float(after["avg_distance"]),
    }


def _global_distance_polish(
    state: IncrementalRepairState,
    candidates,
    cfg: DeepWTASolverConfig,
) -> dict:
    """Deterministic global distance polish.

    Fixed passes, target caps, and per-target route/shot trial caps replace the
    former wall-clock cutoff. Acceptance remains crossing-nonworsening and
    distance-improving by default.
    """
    started = time.perf_counter()
    bucket = cfg.size_bucket(state.N, state.M)
    before = state.tactical_components()

    enabled = (
        cfg.enable_global_distance_polish
        and cfg.distance_polish_passes > 0
    )
    if not enabled:
        return {
            "enabled": False,
            "size_bucket": bucket,
            "passes": 0,
            "moves": 0,
            "route_moves": 0,
            "shot_moves": 0,
            "route_trials": 0,
            "shot_trials": 0,
            "stopped_by_budget": False,
            "runtime_s": float(time.perf_counter() - started),
            "crossing_before": float(before["crossing_count"]),
            "crossing_after": float(before["crossing_count"]),
            "crossing_change": 0.0,
            "avg_distance_before": float(before["avg_distance"]),
            "avg_distance_after": float(before["avg_distance"]),
            "avg_distance_improvement": 0.0,
        }

    moves = 0
    route_moves = 0
    shot_moves = 0
    route_trials = 0
    shot_trials = 0
    passes = 0

    for pass_id in range(cfg.distance_polish_passes):
        burden = np.sum(
            state.route_load.astype(np.float64) * state.distance_metric,
            axis=0,
        )
        targets = np.argsort(-burden, kind="stable")[: cfg.distance_polish_max_targets]
        targets = [int(m) for m in targets.tolist() if burden[m] > 0.0]
        if not targets:
            break

        pass_moves = 0
        for m in targets:
            active_bases = [i for i, target in state.active_routes if target == m]
            active_bases.sort(
                key=lambda i: (-float(state.distance_metric[int(i), m]), int(i))
            )
            if not active_bases:
                continue

            destinations = _all_target_destinations(
                state,
                candidates,
                m,
                exclude=-1,
            )
            destinations.sort(
                key=lambda i: (float(state.distance_metric[int(i), m]), int(i))
            )

            best_route = None
            target_route_trials = 0
            for old_i in active_bases:
                for new_i in destinations:
                    if target_route_trials >= cfg.distance_polish_route_trials_per_target:
                        break
                    new_i = int(new_i)
                    if new_i == int(old_i):
                        continue
                    if state.distance_metric[new_i, m] >= state.distance_metric[int(old_i), m] - cfg.tol:
                        break

                    target_route_trials += 1
                    route_trials += 1
                    effects = state.route_transfer_effects(int(old_i), new_i, m)
                    if effects is None:
                        continue
                    if int(effects["delta_cross"]) > int(cfg.distance_max_crossing_increase):
                        continue
                    if float(effects["delta_avg_distance"]) > -float(cfg.distance_min_avg_improvement):
                        continue
                    if float(effects["delta_cost"]) >= -1e-12:
                        continue

                    key = (
                        float(effects["delta_avg_distance"]),
                        int(effects["delta_cross"]),
                        float(effects["delta_cost"]),
                        int(old_i),
                        new_i,
                    )
                    if best_route is None or key < best_route[0]:
                        best_route = (key, int(old_i), new_i)

                if target_route_trials >= cfg.distance_polish_route_trials_per_target:
                    break

            if best_route is not None:
                _, old_i, new_i = best_route
                state.apply_route_transfer(old_i, new_i, m)
                moves += 1
                route_moves += 1
                pass_moves += 1
                continue

            best_shot = None
            target_shot_trials = 0
            for old_i in active_bases:
                if (int(old_i), m) not in state.active_routes:
                    continue
                for k in np.flatnonzero(state.A[:, int(old_i), m] > 0).tolist():
                    for new_i in destinations:
                        if target_shot_trials >= cfg.distance_polish_shot_trials_per_target:
                            break
                        new_i = int(new_i)
                        if new_i == int(old_i):
                            continue
                        if state.distance_metric[new_i, m] >= state.distance_metric[int(old_i), m] - cfg.tol:
                            break

                        target_shot_trials += 1
                        shot_trials += 1
                        effects = state.bulk_transfer_effects(int(k), int(old_i), new_i, m, 1)
                        if effects is None:
                            continue
                        if int(effects["delta_cross"]) > int(cfg.distance_max_crossing_increase):
                            continue
                        if float(effects["delta_avg_distance"]) > -float(cfg.distance_min_avg_improvement):
                            continue
                        if float(effects["delta_cost"]) >= -1e-12:
                            continue

                        key = (
                            float(effects["delta_avg_distance"]),
                            int(effects["delta_cross"]),
                            float(effects["delta_cost"]),
                            int(k),
                            int(old_i),
                            new_i,
                        )
                        if best_shot is None or key < best_shot[0]:
                            best_shot = (key, int(k), int(old_i), new_i)
                    if target_shot_trials >= cfg.distance_polish_shot_trials_per_target:
                        break
                if target_shot_trials >= cfg.distance_polish_shot_trials_per_target:
                    break

            if best_shot is not None:
                _, k, old_i, new_i = best_shot
                state.apply_bulk_transfer(k, old_i, new_i, m, 1)
                moves += 1
                shot_moves += 1
                pass_moves += 1

        passes = pass_id + 1
        if pass_moves == 0:
            break

    after = state.tactical_components()
    return {
        "enabled": True,
        "size_bucket": bucket,
        "passes": int(passes),
        "moves": int(moves),
        "route_moves": int(route_moves),
        "shot_moves": int(shot_moves),
        "route_trials": int(route_trials),
        "shot_trials": int(shot_trials),
        "stopped_by_budget": False,
        "runtime_s": float(time.perf_counter() - started),
        "crossing_before": float(before["crossing_count"]),
        "crossing_after": float(after["crossing_count"]),
        "crossing_change": float(after["crossing_count"] - before["crossing_count"]),
        "avg_distance_before": float(before["avg_distance"]),
        "avg_distance_after": float(after["avg_distance"]),
        "avg_distance_improvement": float(before["avg_distance"] - after["avg_distance"]),
    }


def _structure_report(
    soft: np.ndarray,
    floor_A: np.ndarray,
    final_A: np.ndarray,
) -> dict[str, float]:
    soft = np.maximum(np.asarray(soft, dtype=np.float64), 0.0)
    floor_A = np.asarray(floor_A, dtype=np.float64)
    final_A = np.asarray(final_A, dtype=np.float64)

    floor_routes = {
        tuple(map(int, x)) for x in np.argwhere(floor_A.sum(axis=0) > 0).tolist()
    }
    final_routes = {
        tuple(map(int, x)) for x in np.argwhere(final_A.sum(axis=0) > 0).tolist()
    }
    intersection = floor_routes & final_routes
    union = floor_routes | final_routes

    final_total = max(float(np.sum(final_A)), 1e-12)
    weighted_retention = float(
        np.sum(np.minimum(soft, final_A)) / final_total
    )
    relative_l1 = float(
        np.sum(np.abs(final_A - soft)) / final_total
    )

    route_mass = soft_route_mass(soft)
    total_route_mass = max(float(np.sum(route_mass)), 1e-12)
    floor_mass = sum(float(route_mass[i, m]) for i, m in floor_routes)
    final_mass = sum(float(route_mass[i, m]) for i, m in final_routes)

    return {
        "structure_floor_route_count": float(len(floor_routes)),
        "structure_final_route_count": float(len(final_routes)),
        "structure_route_retention": float(
            len(intersection) / max(len(final_routes), 1)
        ),
        "structure_route_jaccard": float(
            len(intersection) / max(len(union), 1)
        ),
        "structure_weighted_retention": weighted_retention,
        "structure_relative_l1_to_soft": relative_l1,
        "structure_floor_soft_route_mass_capture": float(
            floor_mass / total_route_mass
        ),
        "structure_final_soft_route_mass_capture": float(
            final_mass / total_route_mass
        ),
        "structure_routes_opened": float(len(final_routes - floor_routes)),
        "structure_routes_closed": float(len(floor_routes - final_routes)),
        "structure_shots_added_vs_floor": float(
            np.sum(np.maximum(final_A - floor_A, 0.0))
        ),
        "structure_shots_removed_vs_floor": float(
            np.sum(np.maximum(floor_A - final_A, 0.0))
        ),
    }


def _seed_inheritance_report(
    seed_A: np.ndarray,
    final_A: np.ndarray,
) -> dict[str, float]:
    """Quantify how much of the final integer solution originates in seed."""
    seed = np.maximum(np.asarray(seed_A, dtype=np.float64), 0.0)
    final = np.maximum(np.asarray(final_A, dtype=np.float64), 0.0)
    inherited = float(np.sum(np.minimum(seed, final)))
    final_shots = float(np.sum(final))
    seed_routes = np.sum(seed, axis=0) > 0
    final_routes = np.sum(final, axis=0) > 0
    inherited_routes = int(np.sum(seed_routes & final_routes))
    final_route_count = int(np.sum(final_routes))
    ratio = inherited / max(final_shots, 1e-12)
    return {
        "seed_final_inheritance_ratio": float(ratio),
        "seed_final_route_inheritance": float(
            inherited_routes / max(final_route_count, 1)
        ),
        "seed_final_inherited_shots": float(inherited),
        "seed_final_inherited_shot_fraction": float(ratio),
        "seed_final_new_routes": float(np.sum(final_routes & ~seed_routes)),
        "seed_final_added_shots": float(np.sum(np.maximum(final - seed, 0.0))),
        "seed_final_removed_shots": float(np.sum(np.maximum(seed - final, 0.0))),
        "seed_final_added_shot_ratio": float(
            np.sum(np.maximum(final - seed, 0.0)) / max(final_shots, 1e-12)
        ),
    }


def _fallback_to_construction(
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
    flight_times,
    cfg,
    reason,
    floor_A,
    seed_A,
    seed_report,
    fill_report,
    seed_scheme_idx,
    started,
):
    from deepwta._feasible_construction import construct_feasible_solution

    A, scheme_idx, old_report = construct_feasible_solution(
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
        return_report=True,
        flight_times=flight_times,
        initial_allocation=seed_A if cfg.seed_preserving_fallback else None,
        initial_floor_allocation=floor_A if cfg.seed_preserving_fallback else None,
        initial_scheme_idx=(
            seed_scheme_idx if cfg.seed_preserving_fallback else None
        ),
    )
    report = dict(old_report)
    report.update(
        {
            "solver_algorithm": STANDARD_ALGORITHM,
            "refinement_enabled": True,
            "construction_fallback_used": True,
            "construction_fallback_reason": str(reason),
            "refinement_wrapper_runtime_s": float(
                time.perf_counter() - started
            ),
            **seed_report,
            "candidate_expansions": int(fill_report.get("candidate_expansions", 0)),
            "residual_fill_shots_added": int(fill_report.get("shots_added", 0)),
            "residual_fill_bulk_operations": int(fill_report.get("bulk_operations", 0)),
            "residual_fill_routes_opened": int(fill_report.get("routes_opened", 0)),
        }
    )
    report.update(_structure_report(assign_intensity, floor_A, A))
    report.update(_seed_inheritance_report(seed_A, A))
    return A, scheme_idx, report


def solve_deepwta(
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
    *,
    config: DeepWTASolverConfig | None = None,
    return_report: bool = False,
    flight_times=None,
):
    """Solve a scenario with DeepWTA-Standard."""
    started = time.perf_counter()
    cfg = config or DeepWTASolverConfig()
    cfg.validate()

    assign_intensity = np.asarray(assign_intensity, dtype=np.float64)
    scheme_probs = np.asarray(scheme_probs, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    G_th = np.asarray(G_th, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    base_coords = np.asarray(base_coords, dtype=np.float64)
    target_coords = np.asarray(target_coords, dtype=np.float64)

    if Q.ndim != 3:
        raise ValueError("Q must have shape [K,N,M]")
    K, N, M = Q.shape
    if assign_intensity.shape != (K, N, M):
        raise ValueError("assign_intensity must have shape [K,N,M]")
    if X.shape != (K, N, M):
        raise ValueError("X must have shape [K,N,M]")
    if B.shape != (K, N):
        raise ValueError("B must have shape [K,N]")
    if U.ndim != 3 or U.shape[1:] != (K, M):
        raise ValueError("U must have shape [L,K,M]")
    L = U.shape[0]
    if scheme_probs.shape != (M, L):
        raise ValueError("scheme_probs must have shape [M,L]")
    if G.shape != (L, M) or G_th.shape != (M,):
        raise ValueError("G/G_th shape mismatch")
    if base_coords.shape != (N, 2) or target_coords.shape != (M, 2):
        raise ValueError("base_coords/target_coords shape mismatch")

    D_norm, _ = _distance_metric(
        base_coords,
        target_coords,
        flight_times,
        K,
        N,
        M,
    )

    scheme_idx = _hard_level_feasible_schemes(
        scheme_probs,
        G,
        G_th,
        cfg.tol,
    )

    selected_demand, _ = selected_plan(U, G, scheme_idx, tol=cfg.tol)
    seed_start = time.perf_counter()
    seed_A, floor_A, residual, seed_report = residual_aware_integerize(
        assign_intensity,
        X,
        B,
        selected_demand,
        Q,
        mode=cfg.integer_seed_mode,
        tol=cfg.tol,
    )
    seed_runtime = time.perf_counter() - seed_start
    seed_report = {
        **seed_report,
        "integer_seed_runtime_s": float(seed_runtime),
        "residual_rounding_runtime_s": float(
            seed_report.get("integer_seed_rounding_kernel_s", 0.0)
        ),
    }

    candidate_start = time.perf_counter()
    candidates = build_target_route_candidates(
        assign_intensity,
        X,
        use_topk=True,
        rho=cfg.route_mass_rho,
        k_min=cfg.route_k_min,
        k_max=cfg.route_k_max,
        tau=cfg.route_softmax_tau,
    )
    candidate_runtime = time.perf_counter() - candidate_start
    initial_k = [
        item.active_count
        for item in candidates.values()
        if len(item.ordered_bases) > 0
    ]
    kmax_hits = sum(
        1
        for item in candidates.values()
        if len(item.ordered_bases) > 0
        and item.active_count >= min(cfg.route_k_max, len(item.ordered_bases))
    )

    seed_routes = {
        tuple(map(int, x))
        for x in np.argwhere(seed_A.sum(axis=0) > 0).tolist()
    }
    conflict_start = time.perf_counter()
    conflict_index = LazyConflictIndex(
        base_coords,
        target_coords,
        initial_active_routes=seed_routes,
    )
    conflict_build_runtime = time.perf_counter() - conflict_start

    weights = TacticalWeights(
        ammo=cfg.weight_ammo,
        crossing=cfg.weight_crossing,
        distance=cfg.weight_distance,
        interception=cfg.weight_interception,
    )
    state = IncrementalRepairState(
        seed_A,
        scheme_idx,
        B,
        U,
        Q,
        X,
        D_norm,
        conflict_index,
        weights=weights,
        tol=cfg.tol,
    )
    floor_components = state.tactical_components()
    floor_cost = state.tactical_cost()

    fill_start = time.perf_counter()
    fill_report = _fill_shortages_route_concentrated(
        state,
        assign_intensity,
        residual,
        candidates,
        D_norm,
        base_coords,
        target_coords,
        cfg,
    )

    # Rare construction rescue: keep the current allocation, switch only the
    # failed target to another damage-level-valid scheme, and retry target-level
    # fill. This is NOT a full target reconstruction.
    scheme_rescue_count = 0
    scheme_rescue_attempts = 0
    if not fill_report["success"] and fill_report["failed_target"] is not None:
        failed_target = int(fill_report["failed_target"])
        current_scheme = int(state.scheme_idx[failed_target])
        valid = np.flatnonzero(G[:, failed_target] >= G_th[failed_target] - cfg.tol)
        alternatives = [int(s) for s in valid.tolist() if int(s) != current_scheme]
        alternatives.sort(
            key=lambda s: (
                float(np.sum(U[s, :, failed_target])),
                -float(scheme_probs[failed_target, s]),
                int(s),
            )
        )

        for alternative in alternatives[:3]:
            scheme_rescue_attempts += 1
            trial = state.clone()
            trial.force_scheme(failed_target, alternative)
            trial_fill = _fill_shortages_route_concentrated(
                trial,
                assign_intensity,
                residual,
                candidates,
                D_norm,
                base_coords,
                target_coords,
                cfg,
            )
            if trial_fill["success"]:
                state.replace_from(trial)
                fill_report = {
                    "success": True,
                    "failed_target": None,
                    "candidate_expansions": int(
                        fill_report["candidate_expansions"]
                        + trial_fill["candidate_expansions"]
                    ),
                    "shots_added": int(
                        fill_report["shots_added"] + trial_fill["shots_added"]
                    ),
                    "bulk_operations": int(
                        fill_report["bulk_operations"] + trial_fill["bulk_operations"]
                    ),
                    "routes_opened": int(
                        fill_report["routes_opened"] + trial_fill["routes_opened"]
                    ),
                    "route_selections": int(
                        fill_report["route_selections"]
                        + trial_fill["route_selections"]
                    ),
                    "active_reuse_selections": int(
                        fill_report["active_reuse_selections"]
                        + trial_fill["active_reuse_selections"]
                    ),
                }
                scheme_rescue_count = 1
                break

    fill_runtime = time.perf_counter() - fill_start

    if not fill_report["success"]:
        if cfg.fallback_to_construction:
            result = _fallback_to_construction(
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
                flight_times,
                cfg,
                f"route_concentrated_fill_failed:target={fill_report['failed_target']}",
                floor_A,
                seed_A,
                seed_report,
                fill_report,
                scheme_idx,
                started,
            )
            return result if return_report else result[:2]
        raise RuntimeError(
            "DeepWTA feasible construction failed for "
            f"target={fill_report['failed_target']}"
        )

    construction_A = state.A.copy()
    construction_scheme = state.scheme_idx.copy()
    construction_components = state.tactical_components()
    construction_cost = state.tactical_cost()

    construction_validation = validate_solution(
        construction_A,
        construction_scheme,
        B,
        U,
        G,
        G_th,
        Q,
        X,
        base_coords,
        target_coords,
        tol=cfg.tol,
    )
    if not bool(construction_validation.get("hard_feasible", False)):
        if cfg.fallback_to_construction:
            result = _fallback_to_construction(
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
                flight_times,
                cfg,
                "post_fill_exact_validation_failed",
                floor_A,
                seed_A,
                seed_report,
                fill_report,
                scheme_idx,
                started,
            )
            return result if return_report else result[:2]
        raise RuntimeError("DeepWTA construction is not hard feasible")

    local_start = time.perf_counter()
    local_report = _local_search(
        state,
        candidates,
        scheme_probs,
        G,
        G_th,
        assign_intensity,
        cfg,
        started,
    )
    local_runtime = time.perf_counter() - local_start

    # Small/medium topology is polished first; then every size receives a
    # crossing-nonworsening distance pass.
    pre_refinement_components = state.tactical_components()
    topology_report = _size_aware_topology_polish(state, candidates, cfg)
    distance_report = _global_distance_polish(state, candidates, cfg)
    post_refinement_components = state.tactical_components()

    validation_start = time.perf_counter()
    final_validation = validate_solution(
        state.A,
        state.scheme_idx,
        B,
        U,
        G,
        G_th,
        Q,
        X,
        base_coords,
        target_coords,
        tol=cfg.tol,
    )
    validation_runtime = time.perf_counter() - validation_start

    if not bool(final_validation.get("hard_feasible", False)):
        if cfg.fallback_to_construction:
            result = _fallback_to_construction(
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
                flight_times,
                cfg,
                "post_local_exact_validation_failed",
                floor_A,
                seed_A,
                seed_report,
                fill_report,
                scheme_idx,
                started,
            )
            return result if return_report else result[:2]
        raise RuntimeError("DeepWTA final solution is not hard feasible")

    final_components = state.tactical_components()
    final_cost = state.tactical_cost()
    raw_scheme_idx = np.argmax(scheme_probs, axis=1).astype(np.int64)

    report = {
        **final_validation,
        "solver_algorithm": STANDARD_ALGORITHM,
        "repair_deterministic": bool(cfg.deterministic_search),
        "refinement_enabled": True,
        "refinement_config": cfg.as_dict(),
        "construction_fallback_used": False,
        "construction_fallback_reason": "",
        "floor_runtime_s": float(seed_runtime),
        "candidate_build_runtime_s": float(candidate_runtime),
        "conflict_build_runtime_s": float(conflict_build_runtime),
        "residual_fill_runtime_s": float(fill_runtime),
        "local_improvement_runtime_s": float(local_runtime),
        "refinement_topology_polish_runtime_s": float(topology_report["runtime_s"]),
        "refinement_distance_polish_runtime_s": float(distance_report["runtime_s"]),
        "refinement_extra_runtime_s": float(
            topology_report["runtime_s"] + distance_report["runtime_s"]
        ),
        "validation_runtime_s": float(validation_runtime),
        "repair_total_runtime_s": float(time.perf_counter() - started),
        # DeepWTA uses Adaptive Top-K; report NaN instead of mislabeling
        # the full reachable-base count as an "initial K" statistic.
        "candidate_mean_initial_k": (
            float(np.mean(initial_k)) if initial_k else float("nan")
        ),
        "candidate_max_initial_k": (
            float(np.max(initial_k)) if initial_k else float("nan")
        ),
        "candidate_kmax_hit_rate": (
            float(kmax_hits / max(len(initial_k), 1))
            if True
            else float("nan")
        ),
        "candidate_expansions": int(fill_report["candidate_expansions"]),
        "residual_fill_shots_added": int(fill_report["shots_added"]),
        "residual_fill_bulk_operations": int(fill_report["bulk_operations"]),
        "residual_fill_routes_opened": int(fill_report["routes_opened"]),
        "residual_fill_route_selections": int(fill_report["route_selections"]),
        "residual_fill_active_reuse_selections": int(
            fill_report["active_reuse_selections"]
        ),
        "scheme_rescue_count": int(scheme_rescue_count),
        "scheme_rescue_attempts": int(scheme_rescue_attempts),
        "construction_hard_feasible": True,
        "construction_tactical_cost": float(construction_cost),
        "construction_crossing_count": float(
            construction_components["crossing_count"]
        ),
        "construction_active_route_count": float(
            np.count_nonzero(construction_A.sum(axis=0) > 0)
        ),
        "final_tactical_cost": float(final_cost),
        "tactical_cost_improvement": float(construction_cost - final_cost),
        "floor_tactical_cost": float(floor_cost),
        "floor_components": floor_components,
        "construction_components": construction_components,
        "final_components": final_components,
        "repair_passes": int(local_report["passes"]),
        "repair_moves": int(
            local_report["moves"]
            + topology_report["moves"]
            + distance_report["moves"]
        ),
        "core_refinement_moves": int(local_report["moves"]),
        "route_consolidation_moves": int(
            local_report["route_consolidation_moves"]
        ),
        "transfer_moves": int(local_report["transfer_moves"]),
        "refinement_size_bucket": str(cfg.size_bucket(N, M)),
        "refinement_pre_refine_crossing_count": float(
            pre_refinement_components["crossing_count"]
        ),
        "refinement_pre_refine_avg_distance": float(
            pre_refinement_components["avg_distance"]
        ),
        "refinement_post_refine_crossing_count": float(
            post_refinement_components["crossing_count"]
        ),
        "refinement_post_refine_avg_distance": float(
            post_refinement_components["avg_distance"]
        ),
        "topology_polish_enabled": bool(topology_report["enabled"]),
        "topology_polish_passes": int(topology_report["passes"]),
        "topology_polish_moves": int(topology_report["moves"]),
        "topology_polish_trials": int(topology_report["trials"]),
        "topology_polish_stopped_by_budget": bool(
            topology_report["stopped_by_budget"]
        ),
        "topology_polish_crossing_reduction": float(
            topology_report["crossing_reduction"]
        ),
        "topology_polish_avg_distance_before": float(
            topology_report["avg_distance_before"]
        ),
        "topology_polish_avg_distance_after": float(
            topology_report["avg_distance_after"]
        ),
        "distance_polish_enabled": bool(distance_report["enabled"]),
        "distance_polish_passes": int(distance_report["passes"]),
        "distance_polish_moves": int(distance_report["moves"]),
        "distance_polish_route_moves": int(distance_report["route_moves"]),
        "distance_polish_shot_moves": int(distance_report["shot_moves"]),
        "distance_polish_route_trials": int(distance_report["route_trials"]),
        "distance_polish_shot_trials": int(distance_report["shot_trials"]),
        "distance_polish_stopped_by_budget": bool(
            distance_report["stopped_by_budget"]
        ),
        "distance_polish_avg_distance_improvement": float(
            distance_report["avg_distance_improvement"]
        ),
        "distance_polish_crossing_change": float(
            distance_report["crossing_change"]
        ),
        "scheme_improvement_moves": int(local_report["scheme_moves"]),
        "stopped_by_budget": bool(local_report["stopped_by_budget"]),
        "stopped_by_stall": bool(local_report["stopped_by_stall"]),
        "raw_scheme_change_count": int(np.sum(raw_scheme_idx != state.scheme_idx)),
        "raw_scheme_change_rate": float(np.mean(raw_scheme_idx != state.scheme_idx)),
        **seed_report,
        **conflict_index.summary(),
    }
    report.update(_structure_report(assign_intensity, floor_A, state.A))
    report.update(_seed_inheritance_report(seed_A, state.A))

    if return_report:
        return state.A.copy(), state.scheme_idx.copy(), report
    return state.A.copy(), state.scheme_idx.copy()
