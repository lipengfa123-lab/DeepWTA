from __future__ import annotations

from collections.abc import Mapping
from typing import Any


METRIC_SCHEMA_COLUMNS = (
    "method",
    "scenario_id",
    "dataset_name",
    "hard_feasible",
    "profile_score",
    "ammo_used",
    "ammo_utilization",
    "crossing_rate",
    "distance",
    "interception",
    "network_time_ms",
    "repair_time_ms",
    "total_runtime_ms",
    "seed",
)


def metric_schema_row(
    source: Mapping[str, Any],
    *,
    dataset_name: str,
    seed: int,
    network_time_ms: float,
    repair_time_ms: float,
    total_runtime_ms: float,
) -> dict[str, Any]:
    """Expose canonical aliases for metrics already computed by exact evaluators."""
    row = {
        "method": source["method"],
        "scenario_id": source["scenario_id"],
        "dataset_name": dataset_name,
        "hard_feasible": source.get("hard_feasible"),
        "profile_score": source.get("profile_score"),
        "ammo_used": source.get("ammo"),
        "ammo_utilization": source.get("mean_inventory_utilization"),
        "crossing_rate": source.get("crossing_pair_rate"),
        "distance": source.get("avg_distance"),
        "interception": source.get("avg_interception"),
        "network_time_ms": float(network_time_ms),
        "repair_time_ms": float(repair_time_ms),
        "total_runtime_ms": float(total_runtime_ms),
        "seed": int(seed),
    }
    return {column: row[column] for column in METRIC_SCHEMA_COLUMNS}


def metric_schema_rows(
    rows: list[Mapping[str, Any]],
    *,
    dataset_name: str,
    default_seed: int,
) -> list[dict[str, Any]]:
    canonical = []
    for row in rows:
        network = float(row.get("gnn_runtime_ms", row.get("network_time_ms", 0.0)))
        total = float(row.get("total_runtime_ms", row.get("runtime_ms", 0.0)))
        group = str(row.get("method_group", ""))
        if group in {"repair", "repair_ablation", "solver"}:
            repair = float(row.get("runtime_ms", 0.0)) if total > network else 0.0
        else:
            repair = 0.0
        canonical.append(
            metric_schema_row(
                row,
                dataset_name=dataset_name,
                seed=int(row.get("seed", row.get("scenario_seed", default_seed))),
                network_time_ms=network,
                repair_time_ms=repair,
                total_runtime_ms=total,
            )
        )
    return canonical
