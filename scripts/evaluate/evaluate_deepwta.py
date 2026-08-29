#!/usr/bin/env python3
"""Evaluate DeepWTA-Standard with the canonical metric schema."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from configs.deepwta_solver_config import DeepWTASolverConfig
from configs.deepwta_train_config import TRAIN_CONFIG
from deepwta.data.manifest import load_dataset_manifest
from deepwta.data.sharded_dataset import ShardedGraphDataset
from deepwta.evaluation.exact_metrics import scenario_metrics
from deepwta.evaluation.metric_schema import METRIC_SCHEMA_COLUMNS, metric_schema_rows
from deepwta.evaluation.profile_objective import attach_profile_reference_metrics, profile_score_breakdown
from deepwta.solver import solve_deepwta
from deepwta.utils.checkpoint_io import build_deepwta_model
from deepwta.utils.numeric import extract_physical, graph_meta_value

METHOD_LABEL = "DeepWTA-Standard"
TUNABLE_SOLVER_FIELDS = {
    "weight_ammo", "weight_crossing", "weight_distance", "weight_interception",
    "route_mass_rho", "route_k_min", "route_k_max", "route_softmax_tau",
    "max_passes", "min_relative_improvement", "stall_passes",
    "max_hotspot_targets", "max_route_consolidation_trials_per_target",
    "max_transfer_trials_per_target", "enable_rare_scheme_change",
    "scheme_probability_gap", "max_scheme_targets",
    "enable_small_medium_topology_polish", "topology_polish_passes",
    "topology_polish_max_targets", "topology_polish_trials_per_target",
    "topology_min_crossing_reduction", "topology_max_avg_distance_increase",
    "enable_global_distance_polish", "distance_polish_passes",
    "distance_polish_max_targets", "distance_polish_route_trials_per_target",
    "distance_polish_shot_trials_per_target", "distance_min_avg_improvement",
    "distance_max_crossing_increase",
}


def parse_indices(text: str | None, dataset_size: int) -> list[int] | None:
    if text is None:
        return None
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--indices must contain unique comma-separated integers")
    if min(values) < 0 or max(values) >= dataset_size:
        raise ValueError(f"--indices outside dataset range [0,{dataset_size - 1}]")
    return values


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reconstruct_allocation(graph, edge_values: torch.Tensor) -> np.ndarray:
    x = graph.x.detach().cpu()
    edge_index = graph.edge_index.detach().cpu()
    base_nodes = torch.nonzero(x[:, 2] < 0.5, as_tuple=False).reshape(-1)
    target_nodes = torch.nonzero(x[:, 2] >= 0.5, as_tuple=False).reshape(-1)
    base_lookup = torch.full((x.size(0),), -1, dtype=torch.long)
    target_lookup = torch.full((x.size(0),), -1, dtype=torch.long)
    base_lookup[base_nodes] = torch.arange(base_nodes.numel())
    target_lookup[target_nodes] = torch.arange(target_nodes.numel())
    base_id = base_lookup[edge_index[0]]
    target_id = target_lookup[edge_index[1]]
    if bool(torch.any(base_id < 0)) or bool(torch.any(target_id < 0)):
        raise ValueError("WTA edge_index must contain base-to-target edges")
    values = edge_values.detach().float().cpu()
    allocation = torch.zeros(
        (int(values.size(1)), int(base_nodes.numel()), int(target_nodes.numel())),
        dtype=values.dtype,
    )
    allocation[:, base_id, target_id] = values.T
    return allocation.numpy()


def solver_config(args, overrides: dict | None = None) -> DeepWTASolverConfig:
    base = DeepWTASolverConfig(
        route_mass_rho=args.rho,
        route_k_min=args.k_min,
        route_k_max=args.k_max,
        route_softmax_tau=args.route_tau,
        max_passes=args.standard_passes,
    )
    values = dict(overrides or {})
    unknown = set(values) - TUNABLE_SOLVER_FIELDS
    if unknown:
        raise ValueError(f"unsupported solver override fields: {sorted(unknown)}")
    config = replace(base, **values)
    config.validate()
    return config


def scalar_report_fields(report: dict) -> dict:
    return {
        f"solver_{key}": value
        for key, value in report.items()
        if isinstance(value, (str, bool, int, float, np.integer, np.floating))
    }


def summarize(frame: pd.DataFrame) -> dict:
    result: dict[str, dict] = {}
    for method, group in frame.groupby("method", sort=False):
        metrics = {}
        for column in (
            "hard_feasible", "profile_score", "ammo_used", "ammo_utilization",
            "crossing_rate", "distance", "interception", "network_time_ms",
            "repair_time_ms", "total_runtime_ms",
        ):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            metrics[column] = {
                "mean": float(np.mean(values)) if len(values) else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        result[str(method)] = {"count": int(len(group)), "metrics": metrics}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument("--indices", default=None, help="Explicit scenario indices for deterministic reruns.")
    parser.add_argument("--tactical-style", default="balanced")
    parser.add_argument("--standard-passes", type=int, default=4)
    parser.add_argument("--rho", type=float, default=0.90)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--route-tau", type=float, default=0.25)
    parser.add_argument(
        "--solver-overrides-json", type=Path, default=None,
        help="Validation-tuning overrides; nondeterministic controls are forbidden.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--warmup-runs", type=int, default=1,
        help="Untimed model warmups before per-scenario runtime measurement.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--solution-output-dir", type=Path, default=None, help="Optional NPZ export for each scenario/mode")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/deepwta_modes")
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    solver_overrides = {}
    if args.solver_overrides_json is not None:
        solver_overrides = json.loads(
            args.solver_overrides_json.read_text(encoding="utf-8")
        )
        if not isinstance(solver_overrides, dict):
            raise ValueError("--solver-overrides-json must contain a JSON object")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_deepwta_model(checkpoint, device)
    manifest = load_dataset_manifest(args.dataset_manifest)
    dataset = ShardedGraphDataset(manifest.root, args.split, TRAIN_CONFIG.shard_cache_size)
    explicit_indices = parse_indices(args.indices, len(dataset))
    if explicit_indices is not None and args.limit != 0:
        raise ValueError("--indices and a nonzero --limit are mutually exclusive")
    indices = (
        explicit_indices
        if explicit_indices is not None
        else list(range(len(dataset) if args.limit == 0 else min(args.limit, len(dataset))))
    )
    count = len(indices)
    output_dir = args.output_root.expanduser().resolve() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if count > 0 and args.warmup_runs:
        warmup_graph = dataset[indices[0]].to(device)
        with torch.no_grad():
            for _ in range(args.warmup_runs):
                model(warmup_graph, scheme_mode="hard", scheme_tau=1.0, gumbel_hard=True)
        sync_device(device)
        del warmup_graph

    detail_rows: list[dict] = []
    for index in tqdm(indices, desc=f"DeepWTA:{args.split}"):
        graph = dataset[index]
        physical = extract_physical(graph)
        scenario_id = str(graph_meta_value(getattr(graph, "scenario_id", None), f"{args.split}_{index:06d}"))
        graph_device = graph.to(device)
        sync_device(device)
        network_started = time.perf_counter()
        with torch.no_grad():
            output = model(graph_device, scheme_mode="hard", scheme_tau=1.0, gumbel_hard=True)
        sync_device(device)
        network_ms = 1000.0 * (time.perf_counter() - network_started)
        allocation_soft = reconstruct_allocation(graph_device, output.allocation)
        scheme_probs = output.scheme_probs.detach().float().cpu().numpy()
        flight_times = physical["D"] / max(float(np.max(physical["D"])), 1e-12)

        for mode in ("standard",):
            method = METHOD_LABEL
            solve_started = time.perf_counter()
            try:
                allocation, schemes, report = solve_deepwta(
                    allocation_soft,
                    scheme_probs,
                    physical["B"], physical["U"], physical["G"], physical["G_th"],
                    physical["Q"], physical["X"], physical["base_coords"], physical["target_coords"],
                    config=solver_config(args, solver_overrides),
                    return_report=True, flight_times=flight_times,
                )
                solver_ms = 1000.0 * (time.perf_counter() - solve_started)
                metrics = scenario_metrics(allocation, schemes, **physical)
                if float(metrics.get("hard_feasible", 0.0)) < 1.0:
                    raise RuntimeError(
                        f"{method} returned a hard-infeasible solution for {scenario_id}"
                    )
                metrics = attach_profile_reference_metrics(
                    metrics, B=physical["B"], U=physical["U"], G=physical["G"],
                    G_th=physical["G_th"], Q=physical["Q"], X=physical["X"],
                )
                profile = profile_score_breakdown(metrics, args.tactical_style)
                if args.solution_output_dir is not None:
                    solution_dir = args.solution_output_dir.expanduser().resolve() / mode
                    solution_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        solution_dir / f"idx_{index:06d}.npz",
                        method=np.asarray(method),
                        instance_id=np.asarray(scenario_id),
                        allocation=np.asarray(allocation, dtype=np.int64),
                        selected_scheme=np.asarray(schemes, dtype=np.int64),
                        base_coords=np.asarray(physical["base_coords"], dtype=np.float64),
                        target_coords=np.asarray(physical["target_coords"], dtype=np.float64),
                        hard_feasible=np.asarray(float(metrics.get("hard_feasible", float("nan")))),
                        profile_score=np.asarray(float(profile.get("profile_score", float("nan")))),
                        ammo_ratio_to_raw_lb=np.asarray(float(metrics.get("ammo_ratio_to_raw_lb", float("nan")))),
                        crossing_pair_rate=np.asarray(float(metrics.get("crossing_pair_rate", float("nan")))),
                    )
                detail_rows.append(
                    {
                        "method": method,
                        "method_group": "solver",
                        "scenario_id": scenario_id,
                        "dataset_name": manifest.dataset_name,
                        "seed": int(args.seed),
                        **metrics,
                        **profile,
                        "network_time_ms": network_ms,
                        "runtime_ms": solver_ms,
                        "total_runtime_ms": network_ms + solver_ms,
                        "solver_mode": mode,
                        "solver_algorithm": report.get("solver_algorithm", "deepwta-standard"),
                        "construction_fallback_used": float(bool(report.get("construction_fallback_used", False))),
                        "error": "",
                        **scalar_report_fields(report),
                    }
                )
            except Exception as exc:
                solver_ms = 1000.0 * (time.perf_counter() - solve_started)
                detail_rows.append(
                    {
                        "method": method, "method_group": "solver", "scenario_id": scenario_id,
                        "dataset_name": manifest.dataset_name, "seed": int(args.seed),
                        "network_time_ms": network_ms, "runtime_ms": solver_ms,
                        "total_runtime_ms": network_ms + solver_ms,
                        "solver_mode": mode, "error": repr(exc),
                    }
                )

    details = pd.DataFrame(detail_rows)
    errors = details["error"].fillna("").astype(str).str.len() > 0
    if errors.any():
        details.to_csv(output_dir / f"{args.split}_details.csv", index=False)
        raise RuntimeError(f"{int(errors.sum())} DeepWTA scenario/mode evaluations failed")
    canonical = pd.DataFrame(
        metric_schema_rows(
            details.to_dict(orient="records"),
            dataset_name=manifest.dataset_name,
            default_seed=args.seed,
        ),
        columns=METRIC_SCHEMA_COLUMNS,
    )
    details.to_csv(output_dir / f"{args.split}_details.csv", index=False)
    canonical.to_csv(output_dir / f"{args.split}_metrics.csv", index=False)
    summary = {
        "run_name": args.run_name,
        "split": args.split,
        "dataset_name": manifest.dataset_name,
        "scenario_count": count,
        "scenario_indices": indices if explicit_indices is not None else None,
        "modes": ["standard"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_run_name": str(checkpoint.get("run_name", "")),
        "checkpoint_training_variant": str(checkpoint.get("training_variant", "")),
        "checkpoint_tactical_style": str(checkpoint.get("tactical_style", "")),
        "warmup_runs": int(args.warmup_runs),
        "solver_overrides": solver_overrides,
        "solver_configs": {
            "standard": solver_config(args, solver_overrides).as_dict()
        },
        "results": summarize(canonical),
    }
    (output_dir / f"{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"metrics : {output_dir / f'{args.split}_metrics.csv'}")
    print(f"summary : {output_dir / f'{args.split}_summary.json'}")


if __name__ == "__main__":
    main()
