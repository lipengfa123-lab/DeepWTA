from __future__ import annotations

"""Zero-crossing feasibility-witness helpers for DeepWTA data generation.

This module is intentionally limited to the *reference feasibility witness*.
It does not change map sampling, Q/X generation, damage-scheme generation,
threshold sampling, inventory-reserve rules, graph features, or the DeepWTA
model.  The special generator mode merely requires the stored witness to be
both hard-feasible and to activate no geometrically crossing base-target route
pairs under the project's canonical exact crossing definition.
"""

from typing import Iterable

import numpy as np

from deepwta.utils.geometry import find_crossing_pairs


def _xy(points_yx: np.ndarray) -> np.ndarray:
    """Convert stored [row(y), col(x)] coordinates to [x,y]."""
    points = np.asarray(points_yx, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape [P,2]")
    return np.stack([points[:, 1], points[:, 0]], axis=1)


def balanced_non_crossing_matching(
    base_points_yx: np.ndarray,
    target_points_yx: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    """Return a non-crossing bichromatic perfect matching for equal-size sets.

    The recursion uses a balanced red-blue dividing segment.  Matching decisions
    are made with an infinitesimal jitter solely to remove grid collinearities;
    the returned edges always reference the original coordinates, and callers
    must perform the canonical exact crossing check on those original points.
    """
    base_points_yx = np.asarray(base_points_yx, dtype=np.float64)
    target_points_yx = np.asarray(target_points_yx, dtype=np.float64)
    if len(base_points_yx) != len(target_points_yx):
        raise ValueError("balanced matching requires equal point counts")
    n = int(len(base_points_yx))
    if n == 0:
        return []

    red = _xy(base_points_yx)
    blue = _xy(target_points_yx)
    span = max(
        float(np.ptp(np.concatenate([red[:, 0], blue[:, 0]]))),
        float(np.ptp(np.concatenate([red[:, 1], blue[:, 1]]))),
        1.0,
    )
    # Only breaks exact grid ties during the combinatorial matching search.
    jitter = span * 1e-9
    red_j = red + rng.normal(0.0, jitter, size=red.shape)
    blue_j = blue + rng.normal(0.0, jitter, size=blue.shape)

    def recurse(red_ids: list[int], blue_ids: list[int]) -> list[tuple[int, int]] | None:
        size = len(red_ids)
        if size == 0:
            return []
        if size == 1:
            return [(int(red_ids[0]), int(blue_ids[0]))]

        # Randomized order gives alternative valid planar matchings across
        # repeated witness attempts while retaining deterministic seed control.
        red_order = list(red_ids)
        rng.shuffle(red_order)
        blue_arr = np.asarray(blue_ids, dtype=np.int64)
        red_arr = np.asarray(red_ids, dtype=np.int64)

        for red_id in red_order:
            origin = red_j[red_id]
            direction = blue_j[blue_arr] - origin  # [B,2]
            red_rel = red_j[red_arr] - origin
            blue_rel = blue_j[blue_arr] - origin

            orient_red = (
                direction[:, 0, None] * red_rel[None, :, 1]
                - direction[:, 1, None] * red_rel[None, :, 0]
            )
            orient_blue = (
                direction[:, 0, None] * blue_rel[None, :, 1]
                - direction[:, 1, None] * blue_rel[None, :, 0]
            )

            red_left = np.sum(orient_red > 0.0, axis=1)
            blue_left = np.sum(orient_blue > 0.0, axis=1)
            candidates = np.flatnonzero(red_left == blue_left)
            if candidates.size > 1:
                candidates = candidates.copy()
                rng.shuffle(candidates)

            for candidate in candidates:
                blue_id = int(blue_arr[int(candidate)])
                row_red = orient_red[int(candidate)]
                row_blue = orient_blue[int(candidate)]

                left_red = [
                    int(item)
                    for pos, item in enumerate(red_ids)
                    if row_red[pos] > 0.0
                ]
                right_red = [
                    int(item)
                    for pos, item in enumerate(red_ids)
                    if item != red_id and row_red[pos] < 0.0
                ]
                left_blue = [
                    int(item)
                    for pos, item in enumerate(blue_ids)
                    if row_blue[pos] > 0.0
                ]
                right_blue = [
                    int(item)
                    for pos, item in enumerate(blue_ids)
                    if item != blue_id and row_blue[pos] < 0.0
                ]

                if len(left_red) != len(left_blue) or len(right_red) != len(right_blue):
                    continue
                left_match = recurse(left_red, left_blue)
                if left_match is None:
                    continue
                right_match = recurse(right_red, right_blue)
                if right_match is None:
                    continue
                return [(int(red_id), int(blue_id))] + left_match + right_match

        return None

    return recurse(list(range(n)), list(range(n)))


def build_zero_crossing_target_base_map(
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Map every target to one base and require the resulting routes to cross 0 times.

    For the formal DeepWTA benchmark scales N=M, this is an exact planar
    bichromatic perfect matching.  Unequal N/M is also supported for custom
    diagnostic data: if N>M a random subset of M bases is matched; if N<M the
    bases are represented by balanced virtual copies.  The final route set is
    always checked on the *original* coordinates with the canonical geometry.
    """
    base_coords = np.asarray(base_coords, dtype=np.float64)
    target_coords = np.asarray(target_coords, dtype=np.float64)
    n = int(len(base_coords))
    m = int(len(target_coords))
    if n <= 0 or m <= 0:
        return None

    if n >= m:
        virtual_base_ids = rng.choice(n, size=m, replace=False).astype(np.int64)
    else:
        repeats, remainder = divmod(m, n)
        chunks = [np.tile(np.arange(n, dtype=np.int64), repeats)] if repeats > 0 else []
        if remainder:
            chunks.append(rng.choice(n, size=remainder, replace=False).astype(np.int64))
        virtual_base_ids = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        rng.shuffle(virtual_base_ids)

    virtual_base_coords = base_coords[virtual_base_ids]
    matching = balanced_non_crossing_matching(virtual_base_coords, target_coords, rng)
    if matching is None:
        return None

    target_to_base = np.full(m, -1, dtype=np.int64)
    for virtual_id, target_id in matching:
        target_to_base[int(target_id)] = int(virtual_base_ids[int(virtual_id)])
    if np.any(target_to_base < 0):
        return None

    routes = [(int(target_to_base[t]), int(t)) for t in range(m)]
    if find_crossing_pairs(base_coords, target_coords, routes):
        return None
    return target_to_base


def _single_base_target_allocation(
    demand: int,
    weapon: int,
    target: int,
    base: int,
    q: np.ndarray,
    x: np.ndarray,
    used: np.ndarray,
    capacity_limit: int,
) -> np.ndarray | None:
    if demand <= 0:
        return np.zeros(used.shape[1], dtype=np.int16)
    if x[weapon, base, target] <= 0.5:
        return None

    free = int(capacity_limit - used[weapon, base])
    if free <= 0:
        return None
    survival = max(1.0 - float(q[weapon, base, target]), 1e-6)
    launches = int(np.ceil(float(demand) / survival - 1e-9))
    if launches <= 0 or launches > free:
        return None

    allocation = np.zeros(used.shape[1], dtype=np.int16)
    allocation[base] = launches
    return allocation


def build_reference_solution_zero_crossing(
    u: np.ndarray,
    g: np.ndarray,
    g_th: np.ndarray,
    resource: np.ndarray,
    q: np.ndarray,
    x: np.ndarray,
    inventory_max: int,
    reference_capacity_ratio: float,
    target_to_base: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Construct the ordinary DeepWTA feasibility witness on a planar route skeleton.

    Candidate scheme ordering is deliberately kept aligned with the original
    generator: valid schemes are sorted by resource, the cheapest top-3 are
    shuffled, and the remainder stay in resource order.  The only additional
    restriction is that every target draws all of its active weapon types from
    its assigned planar base, so the witness activates at most one route per
    target and retains the zero-crossing route skeleton.
    """
    num_schemes, num_weapons, num_targets = u.shape
    num_bases = q.shape[1]
    target_to_base = np.asarray(target_to_base, dtype=np.int64)
    if target_to_base.shape != (num_targets,):
        raise ValueError("target_to_base has an invalid shape")
    if np.any(target_to_base < 0) or np.any(target_to_base >= num_bases):
        raise ValueError("target_to_base contains an invalid base id")

    capacity_limit = max(1, int(np.floor(inventory_max * reference_capacity_ratio)))
    allocation = np.zeros((num_weapons, num_bases, num_targets), dtype=np.int16)
    used = np.zeros((num_weapons, num_bases), dtype=np.int16)
    selected = np.full(num_targets, -1, dtype=np.int16)

    target_order = np.argsort(-g_th.astype(np.float32))
    for target in target_order:
        target = int(target)
        base = int(target_to_base[target])
        valid = np.flatnonzero(g[:, target] >= g_th[target])
        valid = valid[np.argsort(resource[valid, target])]
        top = valid[: min(3, len(valid))].copy()
        if len(top) > 1:
            rng.shuffle(top)
        candidates = np.concatenate([top, valid[min(3, len(valid)):]])

        chosen: int | None = None
        chosen_alloc: np.ndarray | None = None
        for scheme in candidates:
            trial_used = used.copy()
            trial_target = np.zeros((num_weapons, num_bases), dtype=np.int16)
            feasible = True
            for weapon in np.flatnonzero(u[scheme, :, target] > 0):
                weapon = int(weapon)
                alloc = _single_base_target_allocation(
                    int(u[scheme, weapon, target]),
                    weapon,
                    target,
                    base,
                    q,
                    x,
                    trial_used,
                    capacity_limit,
                )
                if alloc is None:
                    feasible = False
                    break
                trial_target[weapon] = alloc
                trial_used[weapon] += alloc
            if feasible:
                chosen = int(scheme)
                chosen_alloc = trial_target
                used = trial_used
                break

        if chosen is None or chosen_alloc is None:
            return None
        selected[target] = chosen
        allocation[:, :, target] = chosen_alloc

    return selected, allocation


def reference_with_inventory_budget_zero_crossing(
    u: np.ndarray,
    g: np.ndarray,
    sampled_thresholds: np.ndarray,
    resource: np.ndarray,
    q: np.ndarray,
    x: np.ndarray,
    inventory_max: int,
    reference_capacity_ratio: float,
    total_weapons: int | None,
    target_to_base: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Zero-crossing counterpart of the current inventory-budget witness logic.

    Threshold handling is intentionally the same as the original v5 generator:
    random-scale data retain sampled thresholds; fixed-inventory benchmarks start
    from level 1 and restore sampled upgrades only when headroom permits.
    """
    if total_weapons is None:
        reference = build_reference_solution_zero_crossing(
            u,
            g,
            sampled_thresholds,
            resource,
            q,
            x,
            inventory_max,
            reference_capacity_ratio,
            target_to_base,
            rng,
        )
        if reference is None:
            return None
        return sampled_thresholds, reference[0], reference[1]

    thresholds = np.ones_like(sampled_thresholds, dtype=np.int8)
    reference = build_reference_solution_zero_crossing(
        u,
        g,
        thresholds,
        resource,
        q,
        x,
        inventory_max,
        reference_capacity_ratio,
        target_to_base,
        rng,
    )
    if reference is None or int(np.sum(reference[1])) > int(total_weapons):
        return None

    selected, allocation = reference
    headroom = max(int(total_weapons) - int(np.sum(allocation)), 0)
    upgrade_targets = np.flatnonzero(sampled_thresholds > 1)
    rng.shuffle(upgrade_targets)
    for target in upgrade_targets:
        current = resource[g[:, target] >= thresholds[target], target]
        desired = resource[g[:, target] >= sampled_thresholds[target], target]
        if current.size == 0 or desired.size == 0:
            continue
        estimated_extra = max(float(np.min(desired) - np.min(current)), 0.0)
        if estimated_extra <= headroom:
            thresholds[target] = sampled_thresholds[target]
            headroom -= int(np.ceil(estimated_extra))

    upgraded = build_reference_solution_zero_crossing(
        u,
        g,
        thresholds,
        resource,
        q,
        x,
        inventory_max,
        reference_capacity_ratio,
        target_to_base,
        rng,
    )
    if upgraded is not None and int(np.sum(upgraded[1])) <= int(total_weapons):
        return thresholds, upgraded[0], upgraded[1]
    return np.ones_like(sampled_thresholds, dtype=np.int8), selected, allocation
