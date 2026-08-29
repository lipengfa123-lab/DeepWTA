#!/usr/bin/env python3
"""Independently validate one exported WTA solution against one case archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from deepwta.benchmark.metrics import evaluate_wta_solution
from deepwta.benchmark.schema import WTAInstance, WTASolution
from deepwta.benchmark.validation import validate_wta_solution


def scalar(array, default):
    return array.item() if array is not None else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--tactical-style", default="balanced")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.data.stat().st_size == 0 or args.result.stat().st_size == 0:
        raise RuntimeError("input and result archives must be non-empty")

    with np.load(args.data, allow_pickle=False) as data, np.load(
        args.result, allow_pickle=False
    ) as result:
        instance = WTAInstance(
            base_coords=data["base_coords"],
            target_coords=data["target_coords"],
            inventory=data["B_inventory"],
            scheme_demand=data["U_scheme_demand"],
            damage_level=data["G_damage_level"],
            damage_threshold=data["G_th_damage_threshold"],
            intercept_prob=data["Q_intercept_prob"],
            distance=data["D_distance"],
            reachable=data["X_reachable"],
            instance_id=str(scalar(data.get("scenario_id"), args.data.stem)),
            seed=int(scalar(data.get("scenario_seed"), 0)),
            dataset_version=str(scalar(data.get("dataset_name"), "wta_e2e_v1")),
        )
        solution = WTASolution(
            selected_scheme=result["selected_scheme"],
            allocation=result["allocation"],
            method=args.result.parent.name,
        )

    report = validate_wta_solution(instance, solution)
    metrics = evaluate_wta_solution(
        instance, solution, tactical_style=args.tactical_style
    )
    output = {
        "scenario_id": instance.instance_id,
        "method": solution.method,
        "feasible": bool(report.get("hard_feasible", False)),
        "inventory_ok": bool(report.get("inventory_feasible", False)),
        "demand_ok": bool(report.get("demand_feasible", False)),
        "damage_ok": bool(report.get("level_feasible", False)),
        "reachability_ok": bool(report.get("reachability_feasible", False)),
        "integrality_ok": bool(report.get("is_integer", False)),
        "J": float(metrics["profile_score"]),
        "Ammo/LB": float(metrics["ammo_ratio_to_raw_lb"]),
        "PCR": float(metrics["crossing_pair_rate"]),
        "average_distance": float(metrics["avg_distance"]),
        "interception_cost": float(metrics["avg_interception"]),
        "crossing_count": int(metrics["crossing_count"]),
    }
    text = json.dumps(output, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
