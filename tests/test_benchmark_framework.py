from __future__ import annotations

from pathlib import Path

import numpy as np

from configs.dataset.benchmark_scales import BENCHMARK_SCALES
from configs.dataset_config import CONFIG
from deepwta.data.inventory import fit_inventory_to_total
from deepwta.data.manifest import load_dataset_manifest, sha256_file
from deepwta.evaluation.metric_schema import METRIC_SCHEMA_COLUMNS, metric_schema_rows


def test_scale_contract_keeps_weapon_type_dimension_frozen() -> None:
    assert CONFIG.wta.num_weapons == 10
    assert {
        name: (scale.total_weapons, scale.targets, scale.bases)
        for name, scale in BENCHMARK_SCALES.items()
    } == {
        "small": (50, 20, 20),
        "medium": (100, 50, 50),
        "large": (200, 100, 100),
        "xlarge": (500, 200, 200),
    }


def test_inventory_budget_is_exact_and_does_not_reduce_reference_use() -> None:
    used = np.array([[2, 0], [1, 1]], dtype=np.int16)
    preferred = np.array([[4, 2], [3, 2]], dtype=np.int16)
    result = fit_inventory_to_total(
        used,
        preferred,
        total_weapons=10,
        inventory_max=8,
        rng=np.random.default_rng(2026),
    )
    assert result is not None
    assert int(result.sum()) == 10
    assert np.all(result >= used)


def test_inventory_budget_rejects_infeasible_total() -> None:
    used = np.array([[3, 2]], dtype=np.int16)
    assert fit_inventory_to_total(
        used,
        used,
        total_weapons=4,
        inventory_max=8,
        rng=np.random.default_rng(1),
    ) is None


def test_manifest_loader_exposes_dataset_identity_and_hash() -> None:
    manifest_path = Path(__file__).with_name("fixtures") / "wta_small.json"
    manifest = load_dataset_manifest(manifest_path)
    assert manifest.dataset_name == "wta_small"
    assert manifest.scale == "small"
    assert manifest.seed == 2026
    assert len(sha256_file(Path("data/maps/Friendly_Map.png"))) == 64


def test_metric_schema_is_alias_only_and_has_stable_order() -> None:
    source = {
        "method": "DeepWTA-Standard",
        "method_group": "solver",
        "scenario_id": "case-1",
        "scenario_seed": 7,
        "seed": 11,
        "hard_feasible": 1.0,
        "profile_score": 2.5,
        "ammo": 12.0,
        "mean_inventory_utilization": 0.4,
        "crossing_pair_rate": 0.1,
        "avg_distance": 0.2,
        "avg_interception": 0.3,
        "gnn_runtime_ms": 4.0,
        "runtime_ms": 6.0,
        "total_runtime_ms": 10.0,
    }
    row = metric_schema_rows([source], dataset_name="wta_small", default_seed=2026)[0]
    assert tuple(row) == METRIC_SCHEMA_COLUMNS
    assert row["ammo_used"] == source["ammo"]
    assert row["ammo_utilization"] == source["mean_inventory_utilization"]
    assert row["crossing_rate"] == source["crossing_pair_rate"]
    assert row["network_time_ms"] == 4.0
    assert row["repair_time_ms"] == 6.0
    assert row["total_runtime_ms"] == 10.0
    assert row["seed"] == 11
