from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.dataset_config import CONFIG
from deepwta.benchmark.adapters import instance_from_graph
from deepwta.data.sharded_dataset import ShardedGraphDataset


def percentile_dict(values: Iterable[float]) -> dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(a.mean()),
        "p10": float(np.quantile(a, 0.10)),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
    }


def witness_ok(inst, tol: float = 1e-7) -> bool:
    if inst.reference_scheme is None or inst.reference_allocation is None:
        return False
    scheme = np.asarray(inst.reference_scheme, dtype=np.int64)
    a = np.asarray(inst.reference_allocation, dtype=np.float64)
    if np.any(scheme < 0) or np.any(scheme >= inst.num_schemes):
        return False
    chosen_level = inst.damage_level[scheme, np.arange(inst.num_targets)]
    if np.any(chosen_level + tol < inst.damage_threshold):
        return False
    if np.any(a < -tol) or np.any(np.abs(a - np.rint(a)) > tol):
        return False
    if np.any(a.sum(axis=2) > inst.inventory + tol):
        return False
    if np.any((a > tol) & (inst.reachable <= 0.5)):
        return False
    demand = inst.scheme_demand[scheme, :, np.arange(inst.num_targets)].T
    effective = np.sum(a * (1.0 - inst.intercept_prob), axis=1)
    return bool(np.all(effective + tol >= demand))


def audit_split(dataset_dir: Path, split: str, limit: int) -> dict:
    ds = ShardedGraphDataset(dataset_dir, split, cache_size=2)
    count = len(ds) if limit <= 0 else min(limit, len(ds))
    seeds: set[int] = set()
    ids: set[str] = set()
    sizes = []
    feasible_scheme_counts = []
    relative_cost_spreads = []
    scheme_cost_cv = []
    max_local_pressure = []
    witness_pass = 0
    no_damage_feasible = 0

    for idx in range(count):
        inst = instance_from_graph(ds[idx], dataset_version=CONFIG.version)
        seeds.add(int(inst.seed)); ids.add(str(inst.instance_id))
        sizes.append((inst.num_bases, inst.num_targets))
        if witness_ok(inst):
            witness_pass += 1

        # Survival-adjusted, target-local optimistic scheme cost.  It is an audit
        # statistic only and is never fed to a learner.
        for j in range(inst.num_targets):
            legal = inst.damage_level[:, j] + 1e-7 >= inst.damage_threshold[j]
            feasible_scheme_counts.append(int(legal.sum()))
            if not np.any(legal):
                no_damage_feasible += 1
                continue
            survival = 1.0 - inst.intercept_prob[:, :, j]  # [K,N]
            reachable = inst.reachable[:, :, j] > 0.5
            best_survival = np.max(np.where(reachable, survival, 0.0), axis=1)
            effective_cap = np.sum(inst.inventory * reachable * survival, axis=1)

            u = inst.scheme_demand[legal, :, j].astype(np.float64)
            cost = np.sum(u / np.maximum(best_survival[None, :], 1e-6), axis=1)
            if cost.size > 1:
                cmin = max(float(cost.min()), 1e-9)
                relative_cost_spreads.append(float((cost.max() - cost.min()) / cmin))
                mean = max(float(cost.mean()), 1e-9)
                scheme_cost_cv.append(float(cost.std() / mean))
            pressure = u / np.maximum(effective_cap[None, :], 1e-9)
            max_local_pressure.extend(np.max(pressure, axis=1).tolist())

    size_arr = np.asarray(sizes, dtype=np.int64)
    return {
        "split": split,
        "count": int(count),
        "unique_seeds": int(len(seeds)),
        "unique_ids": int(len(ids)),
        "seeds": sorted(seeds),
        "ids": sorted(ids),
        "N": percentile_dict(size_arr[:, 0] if count else []),
        "M": percentile_dict(size_arr[:, 1] if count else []),
        "damage_feasible_schemes_per_target": percentile_dict(feasible_scheme_counts),
        "scheme_effective_cost_relative_spread": percentile_dict(relative_cost_spreads),
        "scheme_effective_cost_cv": percentile_dict(scheme_cost_cv),
        "scheme_local_capacity_pressure": percentile_dict(max_local_pressure),
        "targets_without_damage_feasible_scheme": int(no_damage_feasible),
        "witness_feasible_rate": float(witness_pass / max(count, 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Audit a frozen shared E2E WTA dataset before any model training.")
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=0, help="0=all graphs in each split")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    results = {split: audit_split(dataset_dir, split, args.limit) for split in args.splits}

    # Verify split isolation.  The full seed/id lists are removed from the final
    # JSON after overlap checks to keep it compact.
    failures = []
    for a_i, a in enumerate(args.splits):
        for b in args.splits[a_i + 1 :]:
            seed_overlap = set(results[a]["seeds"]) & set(results[b]["seeds"])
            id_overlap = set(results[a]["ids"]) & set(results[b]["ids"])
            if seed_overlap:
                failures.append(f"seed overlap {a}/{b}: {len(seed_overlap)}")
            if id_overlap:
                failures.append(f"scenario-id overlap {a}/{b}: {len(id_overlap)}")

    for split, row in results.items():
        if row["unique_seeds"] != row["count"]:
            failures.append(f"{split}: seeds are not unique")
        if row["unique_ids"] != row["count"]:
            failures.append(f"{split}: scenario ids are not unique")
        if row["targets_without_damage_feasible_scheme"] != 0:
            failures.append(f"{split}: target without damage-feasible scheme")
        if row["witness_feasible_rate"] < 1.0 - 1e-12:
            failures.append(f"{split}: witness feasibility rate={row['witness_feasible_rate']:.6f}")
        row.pop("seeds", None); row.pop("ids", None)

    payload = {
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_dir.name,
        "dataset_version": CONFIG.version,
        "learning_contract": {
            "reference_scheme_is_training_input": False,
            "reference_allocation_is_training_input": False,
            "scheme_hard_mask": "damage_level >= damage_threshold only",
            "same_physical_splits_for_all_methods": True,
        },
        "splits": results,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    output = args.output or (PROJECT_ROOT / "results" / "e2e_dataset_audit" / f"{dataset_dir.name}_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"audit: {output}")
    if failures:
        raise SystemExit("dataset audit failed")


if __name__ == "__main__":
    main()
