from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from deepwta.data.map_data import MapContext, pairwise_distance, sample_coordinates
from deepwta.data.inventory import fit_inventory_to_total
from deepwta.data.zero_crossing import (
    build_zero_crossing_target_base_map,
    reference_with_inventory_budget_zero_crossing,
)
from deepwta.utils.geometry import count_crossings


LEVELS = (1, 2, 3)


def _choose_size(cfg, context: MapContext, rng: np.random.Generator) -> tuple[int, int]:
    """Sample N/M under the formal 1/3 <= N/M <= 3 training distribution."""
    # Fixed benchmark mode for scalability experiments.
    # Training keeps the original random-size sampling below.
    if getattr(cfg, "fixed_targets", None) is not None:
        max_n = len(context.base_candidates)
        max_m = len(context.target_candidates)
        fixed_m = int(cfg.fixed_targets)
        fixed_n = int(cfg.fixed_bases) if getattr(cfg, "fixed_bases", None) is not None else min(max_n, fixed_m)

        if fixed_n > max_n or fixed_m > max_m:
            raise RuntimeError(
                f"fixed scale exceeds available map candidates: "
                f"bases={fixed_n}/{max_n}, targets={fixed_m}/{max_m}"
            )

        if not (cfg.min_size_ratio <= fixed_n / fixed_m <= cfg.max_size_ratio):
            raise RuntimeError(
                f"fixed N/M ratio violates constraints: {fixed_n}/{fixed_m}"
            )

        return fixed_n, fixed_m

    max_n = min(cfg.max_bases, len(context.base_candidates))
    max_m = min(cfg.max_targets, len(context.target_candidates))

    valid_typical = [
        (n, m)
        for n, m in cfg.typical_sizes
        if n <= max_n
        and m <= max_m
        and cfg.min_size_ratio <= (n / m) <= cfg.max_size_ratio
    ]
    if valid_typical and rng.random() >= cfg.random_size_ratio:
        return valid_typical[int(rng.integers(0, len(valid_typical)))]

    for _ in range(64):
        n = int(rng.integers(cfg.min_bases, max_n + 1))
        min_m = max(cfg.min_targets, int(np.ceil(n / cfg.max_size_ratio)))
        max_m_for_n = min(max_m, int(np.floor(n / cfg.min_size_ratio)))
        if min_m <= max_m_for_n:
            m = int(rng.integers(min_m, max_m_for_n + 1))
            return n, m

    raise RuntimeError("cannot sample a valid N/M pair under the configured ratio bounds")


def _generate_intercept_prob(
    num_weapons: int,
    num_bases: int,
    num_targets: int,
    distance_norm: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    penetration = rng.beta(2.5, 2.0, size=(num_weapons, 1, 1))
    base_exposure = rng.beta(2.0, 3.0, size=(1, num_bases, 1))
    target_defense = rng.beta(2.2, 2.0, size=(1, 1, num_targets))
    noise = rng.normal(0.0, 0.012, size=(num_weapons, num_bases, num_targets))

    q = (
        0.08
        + 0.12 * target_defense
        + 0.05 * base_exposure
        + 0.07 * distance_norm[None, :, :]
        - 0.10 * penetration
        + noise
    )
    return np.clip(q, 0.03, 0.35).astype(np.float32)


def _generate_reachability(
    num_weapons: int,
    num_bases: int,
    num_targets: int,
    q: np.ndarray,
    distance_norm: np.ndarray,
    target_rate: float,
    min_reachable_bases: int,
    rng: np.random.Generator,
) -> np.ndarray:
    weapon_bias = rng.normal(0.0, 0.008, size=(num_weapons, 1, 1))
    base_bias = rng.normal(0.0, 0.006, size=(1, num_bases, 1))
    target_bias = rng.normal(0.0, 0.008, size=(1, 1, num_targets))
    probability = np.clip(target_rate + weapon_bias + base_bias + target_bias, 0.90, 0.995)
    x = (rng.random((num_weapons, num_bases, num_targets)) < probability).astype(np.float32)

    required = min(max(1, min_reachable_bases), num_bases)
    for k in range(num_weapons):
        for m in range(num_targets):
            current = np.flatnonzero(x[k, :, m] > 0.5)
            if len(current) >= required:
                continue
            # Prefer low-interception and short routes when restoring connectivity.
            score = q[k, :, m] + 0.12 * distance_norm[:, m]
            order = np.argsort(score)
            x[k, order[:required], m] = 1.0
    return x


def _mean_reachable_q(q: np.ndarray, x: np.ndarray, target: int) -> np.ndarray:
    num_weapons = q.shape[0]
    result = np.empty(num_weapons, dtype=np.float32)
    for k in range(num_weapons):
        mask = x[k, :, target] > 0.5
        result[k] = float(np.mean(q[k, mask, target])) if np.any(mask) else 0.35
    return result


def _scheme_resource(demand: np.ndarray, mean_q: np.ndarray) -> float:
    return float(np.sum(demand / np.maximum(1.0 - mean_q, 1e-6)))


def _level_specs(level: int, num_weapons: int, demand_max: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if level == 1:
        return (1, min(2, num_weapons)), (1, min(3, demand_max))
    if level == 2:
        return (2 if num_weapons >= 2 else 1, min(3, num_weapons)), (2, min(5, demand_max))
    return (2 if num_weapons >= 2 else 1, min(4, num_weapons)), (max(3, min(4, demand_max)), demand_max)


def _sample_scheme(
    level: int,
    weapon_pool: np.ndarray,
    num_weapons: int,
    demand_max: int,
    rng: np.random.Generator,
) -> np.ndarray:
    type_range, amount_range = _level_specs(level, num_weapons, demand_max)
    low_types, high_types = type_range
    active_count = int(rng.integers(low_types, high_types + 1))
    active = rng.choice(weapon_pool, size=min(active_count, len(weapon_pool)), replace=False)

    demand = np.zeros(num_weapons, dtype=np.int16)
    low_amount, high_amount = amount_range
    demand[active] = rng.integers(low_amount, high_amount + 1, size=len(active), dtype=np.int16)
    return demand


def _raise_group_resource(
    schemes: list[np.ndarray],
    resources: np.ndarray,
    levels: np.ndarray,
    target_level: int,
    required_mean: float,
    mean_q: np.ndarray,
    demand_max: int,
    rng: np.random.Generator,
) -> bool:
    indices = np.flatnonzero(levels == target_level)
    if len(indices) == 0:
        return False

    for _ in range(256):
        current_mean = float(np.mean(resources[indices]))
        if current_mean >= required_mean:
            return True

        idx = int(rng.choice(indices))
        demand = schemes[idx]
        candidates = np.flatnonzero(demand > 0)
        if len(candidates) == 0:
            return False
        k = int(rng.choice(candidates))
        if demand[k] < demand_max:
            demand[k] += 1
            resources[idx] = _scheme_resource(demand, mean_q)
        else:
            expandable = np.flatnonzero(demand < demand_max)
            if len(expandable) == 0:
                continue
            k = int(rng.choice(expandable))
            demand[k] = max(1, demand[k])
            resources[idx] = _scheme_resource(demand, mean_q)
    return False


def _generate_damage_schemes(
    num_schemes: int,
    num_weapons: int,
    num_targets: int,
    demand_max: int,
    q: np.ndarray,
    x: np.ndarray,
    level_resource_gap: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_schemes < 3:
        raise ValueError("num_schemes must be at least 3 to include Light/Medium/Heavy")

    u = np.zeros((num_schemes, num_weapons, num_targets), dtype=np.int16)
    g = np.zeros((num_schemes, num_targets), dtype=np.int8)
    resource = np.zeros((num_schemes, num_targets), dtype=np.float32)

    extra_probs = np.array([0.35, 0.40, 0.25], dtype=np.float64)
    for m in range(num_targets):
        levels = np.array([1, 2, 3] + rng.choice(LEVELS, size=num_schemes - 3, p=extra_probs).tolist(), dtype=np.int8)
        rng.shuffle(levels)
        g[:, m] = levels

        pool_size = min(num_weapons, max(4, int(round(num_weapons * 0.65))))
        weapon_pool = rng.choice(num_weapons, size=pool_size, replace=False)
        mean_q = _mean_reachable_q(q, x, m)

        schemes: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        resources = np.zeros(num_schemes, dtype=np.float32)

        for l, level in enumerate(levels):
            demand = None
            for _ in range(64):
                candidate = _sample_scheme(int(level), weapon_pool, num_weapons, demand_max, rng)
                key = tuple(int(v) for v in candidate)
                if key not in seen:
                    demand = candidate
                    seen.add(key)
                    break
            if demand is None:
                demand = _sample_scheme(int(level), np.arange(num_weapons), num_weapons, demand_max, rng)

            schemes.append(demand)
            resources[l] = _scheme_resource(demand, mean_q)

        light = np.flatnonzero(levels == 1)
        medium = np.flatnonzero(levels == 2)
        heavy = np.flatnonzero(levels == 3)
        light_mean = float(np.mean(resources[light]))

        if not _raise_group_resource(
            schemes, resources, levels, 2,
            light_mean * (1.0 + level_resource_gap), mean_q, demand_max, rng,
        ):
            raise RuntimeError("cannot enforce Medium resource hierarchy")

        medium_mean = float(np.mean(resources[medium]))
        if not _raise_group_resource(
            schemes, resources, levels, 3,
            medium_mean * (1.0 + level_resource_gap), mean_q, demand_max, rng,
        ):
            raise RuntimeError("cannot enforce Heavy resource hierarchy")

        for l, demand in enumerate(schemes):
            u[l, :, m] = demand
            resource[l, m] = _scheme_resource(demand, mean_q)

    return u, g, resource


def _allocate_expected_demand(
    demand: int,
    weapon: int,
    target: int,
    q: np.ndarray,
    x: np.ndarray,
    distance_norm: np.ndarray,
    used: np.ndarray,
    capacity_limit: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if demand <= 0:
        return np.zeros(used.shape[1], dtype=np.int16)

    reachable = np.flatnonzero(x[weapon, :, target] > 0.5)
    if len(reachable) == 0:
        return None

    load_ratio = used[weapon, reachable] / max(capacity_limit, 1)
    score = (
        q[weapon, reachable, target]
        + 0.12 * distance_norm[reachable, target]
        + 0.18 * load_ratio
        + rng.normal(0.0, 0.01, size=len(reachable))
    )
    order = reachable[np.argsort(score)]

    remaining = float(demand)
    allocation = np.zeros(used.shape[1], dtype=np.int16)
    for base in order:
        free = int(capacity_limit - used[weapon, base])
        if free <= 0:
            continue

        survival = max(1.0 - float(q[weapon, base, target]), 1e-6)
        required_launches = int(np.ceil(remaining / survival - 1e-9))
        launches = min(free, required_launches)
        if launches <= 0:
            continue

        allocation[base] += launches
        remaining -= launches * survival
        if remaining <= 1e-6:
            return allocation
    return None


def _build_reference_solution(
    u: np.ndarray,
    g: np.ndarray,
    g_th: np.ndarray,
    resource: np.ndarray,
    q: np.ndarray,
    x: np.ndarray,
    distance_norm: np.ndarray,
    inventory_max: int,
    reference_capacity_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    num_schemes, num_weapons, num_targets = u.shape
    num_bases = q.shape[1]
    capacity_limit = max(1, int(np.floor(inventory_max * reference_capacity_ratio)))

    allocation = np.zeros((num_weapons, num_bases, num_targets), dtype=np.int16)
    used = np.zeros((num_weapons, num_bases), dtype=np.int16)
    selected = np.full(num_targets, -1, dtype=np.int16)

    target_order = np.argsort(-g_th.astype(np.float32))
    for m in target_order:
        valid = np.flatnonzero(g[:, m] >= g_th[m])
        valid = valid[np.argsort(resource[valid, m])]
        top = valid[: min(3, len(valid))].copy()
        if len(top) > 1:
            rng.shuffle(top)
        candidates = np.concatenate([top, valid[min(3, len(valid)):]])

        chosen = None
        chosen_alloc = None
        for scheme in candidates:
            trial_used = used.copy()
            trial_target = np.zeros((num_weapons, num_bases), dtype=np.int16)
            feasible = True
            for k in np.flatnonzero(u[scheme, :, m] > 0):
                alloc = _allocate_expected_demand(
                    int(u[scheme, k, m]), int(k), int(m), q, x, distance_norm,
                    trial_used, capacity_limit, rng,
                )
                if alloc is None:
                    feasible = False
                    break
                trial_target[k] = alloc
                trial_used[k] += alloc
            if feasible:
                chosen = int(scheme)
                chosen_alloc = trial_target
                used = trial_used
                break

        if chosen is None or chosen_alloc is None:
            return None

        selected[m] = chosen
        allocation[:, :, m] = chosen_alloc

    return selected, allocation


def _generate_inventory(
    allocation_ref: np.ndarray,
    inventory_max: int,
    reserve_rate_min: float,
    reserve_rate_max: float,
    idle_lambda_min: float,
    idle_lambda_max: float,
    rng: np.random.Generator,
    total_weapons: int | None = None,
) -> np.ndarray | None:
    used = np.sum(allocation_ref, axis=2).astype(np.int16)
    num_weapons, num_bases = used.shape

    weapon_supply = rng.beta(2.2, 2.0, size=(num_weapons, 1))
    base_capacity = rng.beta(2.0, 2.2, size=(1, num_bases))
    local_noise = rng.normal(0.0, 0.05, size=(num_weapons, num_bases))

    blend = 0.55 * weapon_supply + 0.45 * base_capacity + local_noise
    blend = np.clip(blend, 0.0, 1.0)
    reserve_rate = reserve_rate_min + (reserve_rate_max - reserve_rate_min) * blend

    usage_reserve = rng.poisson(np.maximum(used * reserve_rate * 0.60, 0.0)).astype(np.int16)
    idle_lambda = idle_lambda_min + (idle_lambda_max - idle_lambda_min) * (
        0.55 * weapon_supply + 0.45 * base_capacity
    )
    idle_inventory = rng.poisson(idle_lambda).astype(np.int16)

    inventory = used + usage_reserve + idle_inventory
    inventory = np.minimum(inventory, inventory_max).astype(np.int16)

    if total_weapons is not None:
        inventory = fit_inventory_to_total(
            used,
            inventory,
            total_weapons=int(total_weapons),
            inventory_max=inventory_max,
            rng=rng,
        )
        if inventory is None:
            return None
    if np.any(inventory < used):
        raise RuntimeError("inventory generation violated the reference allocation")
    return inventory


def _reference_with_inventory_budget(
    u: np.ndarray,
    g: np.ndarray,
    sampled_thresholds: np.ndarray,
    resource: np.ndarray,
    q: np.ndarray,
    x: np.ndarray,
    distance_norm: np.ndarray,
    inventory_max: int,
    reference_capacity_ratio: float,
    total_weapons: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build a feasible witness while respecting an optional global inventory.

    Random training keeps the original sampled thresholds. Fixed benchmarks
    start from level 1 and restore sampled level upgrades only when the exact
    total inventory budget can still support the resulting witness.
    """
    if total_weapons is None:
        reference = _build_reference_solution(
            u, g, sampled_thresholds, resource, q, x, distance_norm,
            inventory_max, reference_capacity_ratio, rng,
        )
        if reference is None:
            return None
        return sampled_thresholds, reference[0], reference[1]

    thresholds = np.ones_like(sampled_thresholds, dtype=np.int8)
    reference = _build_reference_solution(
        u, g, thresholds, resource, q, x, distance_norm,
        inventory_max, reference_capacity_ratio, rng,
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

    upgraded = _build_reference_solution(
        u, g, thresholds, resource, q, x, distance_norm,
        inventory_max, reference_capacity_ratio, rng,
    )
    if upgraded is not None and int(np.sum(upgraded[1])) <= int(total_weapons):
        return thresholds, upgraded[0], upgraded[1]
    return np.ones_like(sampled_thresholds, dtype=np.int8), selected, allocation


def _validate_scenario(
    scenario: dict[str, np.ndarray],
    level_resource_gap: float,
    inventory_max: int,
    require_zero_crossing: bool = False,
) -> None:
    b = scenario["B"]
    u = scenario["U"]
    g = scenario["G"]
    g_th = scenario["G_th"]
    q = scenario["Q"]
    x = scenario["X"]
    d = scenario["D"]
    base_coords = scenario["base_coords"]
    target_coords = scenario["target_coords"]
    selected = scenario["reference_scheme"]
    a_ref = scenario["reference_allocation"]
    resource = scenario["scheme_resource"]

    k, n, m = q.shape
    l = u.shape[0]
    expected = {
        "B": (k, n), "U": (l, k, m), "G": (l, m), "G_th": (m,),
        "X": (k, n, m), "D": (n, m), "base_coords": (n, 2), "target_coords": (m, 2),
        "reference_scheme": (m,), "reference_allocation": (k, n, m),
    }
    for name, shape in expected.items():
        if scenario[name].shape != shape:
            raise ValueError(f"{name} shape {scenario[name].shape} != {shape}")

    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("Q must be in [0,1]")
    if np.any((x != 0.0) & (x != 1.0)):
        raise ValueError("X must be binary")
    if np.any(b < 0) or np.any(b > inventory_max):
        raise ValueError("B exceeds its configured range")
    if np.any(u < 0):
        raise ValueError("U must be non-negative")

    for target in range(m):
        levels = set(int(v) for v in g[:, target])
        if not {1, 2, 3}.issubset(levels):
            raise ValueError("each target must include Light/Medium/Heavy schemes")
        means = [float(np.mean(resource[g[:, target] == level, target])) for level in LEVELS]
        if means[1] + 1e-6 < means[0] * (1.0 + level_resource_gap):
            raise ValueError("Medium expected resource is too small")
        if means[2] + 1e-6 < means[1] * (1.0 + level_resource_gap):
            raise ValueError("Heavy expected resource is too small")

    if np.any(g[selected, np.arange(m)] < g_th):
        raise ValueError("reference scheme violates G_th")
    if np.any(a_ref < 0) or not np.allclose(a_ref, np.rint(a_ref)):
        raise ValueError("reference allocation must be non-negative integer")
    if np.any(np.sum(a_ref, axis=2) > b):
        raise ValueError("reference allocation exceeds inventory")
    if np.any((a_ref > 0) & (x < 0.5)):
        raise ValueError("reference allocation uses unreachable routes")

    chosen_demand = u[selected, :, np.arange(m)].T
    effective_hits = np.sum(a_ref * (1.0 - q), axis=1)
    if np.any(effective_hits + 1e-6 < chosen_demand):
        raise ValueError("reference allocation does not satisfy post-interception demand")

    crossing_count = count_crossings(a_ref, base_coords, target_coords)
    if require_zero_crossing and crossing_count != 0:
        raise ValueError(
            f"zero-crossing witness mode requires C(A_ref)=0, got {crossing_count}"
        )

    recomputed_d = pairwise_distance(base_coords, target_coords)
    if not np.allclose(d, recomputed_d, atol=1e-4):
        raise ValueError("D is inconsistent with coordinates")


def build_wta_graph(scenario: dict[str, Any], context: MapContext, cfg) -> Data:
    b = scenario["B"]
    u = scenario["U"]
    g = scenario["G"]
    g_th = scenario["G_th"]
    q = scenario["Q"]
    x_mask = scenario["X"]
    d = scenario["D"]
    base_coords = scenario["base_coords"]
    target_coords = scenario["target_coords"]

    num_weapons, num_bases = b.shape
    num_schemes, _, num_targets = u.shape

    # base_coords/target_coords are stored [row(y), col(x)] for OpenCV compatibility.
    base_xy = np.stack([base_coords[:, 1] / context.width, base_coords[:, 0] / context.height], axis=1)
    target_xy = np.stack([target_coords[:, 1] / context.width, target_coords[:, 0] / context.height], axis=1)

    inventory_norm = b.astype(np.float32) / max(float(cfg.inventory_max), 1.0)
    demand_norm = u.astype(np.float32) / max(float(cfg.demand_max), 1.0)
    damage_norm = g.astype(np.float32) / 3.0
    threshold_norm = g_th.astype(np.float32) / 3.0

    node_dim = 4 + num_weapons + num_schemes * num_weapons + num_schemes
    base_features = np.zeros((num_bases, node_dim), dtype=np.float32)
    target_features = np.zeros((num_targets, node_dim), dtype=np.float32)

    base_features[:, 0:2] = base_xy
    base_features[:, 2] = 0.0
    base_features[:, 3 : 3 + num_weapons] = inventory_norm.T

    target_features[:, 0:2] = target_xy
    target_features[:, 2] = 1.0
    offset = 3 + num_weapons
    target_features[:, offset : offset + num_schemes * num_weapons] = (
        demand_norm.transpose(2, 0, 1).reshape(num_targets, -1)
    )
    offset += num_schemes * num_weapons
    target_features[:, offset : offset + num_schemes] = damage_norm.T
    target_features[:, -1] = threshold_norm

    src = np.repeat(np.arange(num_bases, dtype=np.int64), num_targets)
    dst = np.tile(np.arange(num_targets, dtype=np.int64) + num_bases, num_bases)
    edge_index = np.stack([src, dst], axis=0)

    d_norm = (d / context.max_distance).reshape(-1, 1).astype(np.float32)
    q_flat = q.transpose(1, 2, 0).reshape(num_bases * num_targets, num_weapons)
    x_flat = x_mask.transpose(1, 2, 0).reshape(num_bases * num_targets, num_weapons)
    edge_attr = np.concatenate([d_norm, q_flat, x_flat], axis=1).astype(np.float32)

    data = Data(
        x=torch.from_numpy(np.concatenate([base_features, target_features], axis=0)),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
    )

    # Raw physical tensors retained for loss, repair, validation and reproducibility.
    data.u = torch.from_numpy(u.astype(np.float32))
    data.B = torch.from_numpy(b.astype(np.float32))
    data.Q = torch.from_numpy(q.astype(np.float32))
    data.U = torch.from_numpy(u.astype(np.float32))
    data.G = torch.from_numpy(g.astype(np.float32))
    data.G_th = torch.from_numpy(g_th.astype(np.float32))
    data.X = torch.from_numpy(x_mask.astype(np.float32))
    data.D = torch.from_numpy(d.astype(np.float32))
    data.base_coords = torch.from_numpy(base_coords.astype(np.float32))
    data.target_coords = torch.from_numpy(target_coords.astype(np.float32))
    data.reference_scheme = torch.from_numpy(scenario["reference_scheme"].astype(np.int64))
    data.reference_allocation = torch.from_numpy(scenario["reference_allocation"].astype(np.float32))
    data.scheme_resource = torch.from_numpy(scenario["scheme_resource"].astype(np.float32))
    data.num_bases = int(num_bases)
    data.num_targets = int(num_targets)
    data.num_weapons = int(num_weapons)
    data.num_schemes = int(num_schemes)
    data.total_weapons = int(np.sum(b))
    data.benchmark_scale = str(getattr(cfg, "benchmark_scale", "") or "")
    data.scenario_seed = int(scenario["scenario_seed"])
    data.scenario_id = str(scenario["scenario_id"])
    data.reference_crossing_count = int(
        scenario.get(
            "reference_crossing_count",
            count_crossings(
                scenario["reference_allocation"],
                scenario["base_coords"],
                scenario["target_coords"],
            ),
        )
    )
    data.zero_crossing_witness = bool(
        scenario.get("zero_crossing_witness", False)
    )
    return data


def generate_wta_graph(cfg, context: MapContext, seed: int, scenario_id: str) -> Data:
    rng = np.random.default_rng(seed)

    for _ in range(cfg.max_scenario_attempts):
        num_bases, num_targets = _choose_size(cfg, context, rng)
        base_coords = sample_coordinates(context.base_candidates, num_bases, rng)
        target_coords = sample_coordinates(context.target_candidates, num_targets, rng)

        d = pairwise_distance(base_coords, target_coords)
        d_norm = d / max(context.max_distance, 1e-6)
        q = _generate_intercept_prob(cfg.num_weapons, num_bases, num_targets, d_norm, rng)
        x = _generate_reachability(
            cfg.num_weapons, num_bases, num_targets, q, d_norm,
            cfg.reachability_rate, cfg.min_reachable_bases, rng,
        )

        try:
            u, g, resource = _generate_damage_schemes(
                cfg.num_schemes, cfg.num_weapons, num_targets, cfg.demand_max,
                q, x, cfg.level_resource_gap, rng,
            )
        except RuntimeError:
            continue

        g_th = rng.choice(
            np.array([1, 2, 3], dtype=np.int8),
            size=num_targets,
            p=np.array(cfg.damage_threshold_probs, dtype=np.float64),
        )

        require_zero_crossing = bool(
            getattr(cfg, "require_zero_crossing_witness", False)
        )
        reference = None
        if require_zero_crossing:
            matching_attempts = max(
                1, int(getattr(cfg, "zero_crossing_matching_attempts", 32))
            )
            for _matching_try in range(matching_attempts):
                target_to_base = build_zero_crossing_target_base_map(
                    base_coords, target_coords, rng
                )
                if target_to_base is None:
                    continue
                reference = reference_with_inventory_budget_zero_crossing(
                    u, g, g_th, resource, q, x,
                    cfg.inventory_max, cfg.reference_capacity_ratio,
                    getattr(cfg, "total_weapons", None),
                    target_to_base, rng,
                )
                if reference is None:
                    continue
                candidate_thresholds, candidate_selected, candidate_a_ref = reference
                if count_crossings(candidate_a_ref, base_coords, target_coords) == 0:
                    reference = (
                        candidate_thresholds, candidate_selected, candidate_a_ref
                    )
                    break
                reference = None
        else:
            reference = _reference_with_inventory_budget(
                u, g, g_th, resource, q, x, d_norm,
                cfg.inventory_max, cfg.reference_capacity_ratio,
                getattr(cfg, "total_weapons", None), rng,
            )

        if reference is None:
            continue
        g_th, selected, a_ref = reference

        b = _generate_inventory(
            a_ref, cfg.inventory_max,
            cfg.reserve_rate_min, cfg.reserve_rate_max,
            cfg.idle_inventory_lambda_min, cfg.idle_inventory_lambda_max,
            rng, getattr(cfg, "total_weapons", None),
        )
        if b is None:
            continue

        scenario = {
            "B": b,
            "Q": q,
            "U": u,
            "G": g,
            "G_th": g_th,
            "X": x,
            "D": d,
            "base_coords": base_coords,
            "target_coords": target_coords,
            "reference_scheme": selected,
            "reference_allocation": a_ref,
            "scheme_resource": resource,
            "scenario_seed": int(seed),
            "scenario_id": str(scenario_id),
            "reference_crossing_count": int(
                count_crossings(a_ref, base_coords, target_coords)
            ),
            "zero_crossing_witness": bool(require_zero_crossing),
        }
        try:
            _validate_scenario(
                scenario,
                cfg.level_resource_gap,
                cfg.inventory_max,
                require_zero_crossing=require_zero_crossing,
            )
        except ValueError:
            continue
        return build_wta_graph(scenario, context, cfg)

    raise RuntimeError(f"failed to generate a valid scenario after {cfg.max_scenario_attempts} attempts")
