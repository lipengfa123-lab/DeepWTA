#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from configs.deepwta_solver_config import DeepWTASolverConfig
from configs.deepwta_train_config import TRAIN_CONFIG
from deepwta.data.manifest import load_dataset_manifest
from deepwta.data.map_data import load_land_mask
from deepwta.data.sharded_dataset import ShardedGraphDataset
from deepwta.evaluation.exact_metrics import scenario_metrics
from deepwta.evaluation.profile_objective import (
    attach_profile_reference_metrics,
    profile_score_breakdown,
)
from deepwta.solver import solve_deepwta
from deepwta.utils.checkpoint_io import build_deepwta_model
from deepwta.utils.geometry import crossing_pairs_from_allocations
from deepwta.utils.numeric import extract_physical, graph_meta_value
from deepwta.utils.paths import project_path


FRIENDLY_COLOR = "#ff6b63"
TARGET_COLOR = "#4aa8e8"
TRAJECTORY_COLOR = "#e6eaed"
CROSSING_ROUTE_COLOR = "#ffd166"
ACTIVE_TARGET_HALO = "#d7c85f"
DARK_BG = "#0e1b25"
DARK_GRID = "#29404d"


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _portable_project_path(value: str | Path) -> Path:
    text = str(value).replace("\\", os.sep).replace("/", os.sep)
    return project_path(text)


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
    dense = torch.zeros(
        (int(values.size(1)), int(base_nodes.numel()), int(target_nodes.numel())),
        dtype=values.dtype,
    )
    dense[:, base_id, target_id] = values.T
    return dense.numpy()


def build_solver_config(args: argparse.Namespace) -> DeepWTASolverConfig:
    cfg = DeepWTASolverConfig(
        route_mass_rho=args.rho,
        route_k_min=args.k_min,
        route_k_max=args.k_max,
        route_softmax_tau=args.route_tau,
        max_passes=args.standard_passes,
        fast_path_max_fill_ratio=args.fast_max_fill_ratio,
        fast_mode=(args.mode == "fast"),
    )
    cfg.validate()
    return cfg


def infer_map_paths(manifest) -> tuple[Path, Path]:
    map_cfg = manifest.payload.get("map") or manifest.payload.get("generation_config", {}).get("map", {})
    friendly = map_cfg.get("friendly_map")
    enemy = map_cfg.get("enemy_map")
    if not friendly or not enemy:
        friendly = "data/maps/Friendly_Map.png"
        enemy = "data/maps/Enemy_Map.png"
    friendly_path = _portable_project_path(friendly)
    enemy_path = _portable_project_path(enemy)
    if not friendly_path.is_file():
        raise FileNotFoundError(f"friendly map not found: {friendly_path}")
    if not enemy_path.is_file():
        raise FileNotFoundError(f"enemy map not found: {enemy_path}")
    return friendly_path, enemy_path


def load_merged_satellite(friendly_path: Path, enemy_path: Path) -> np.ndarray:
    friendly = cv2.imread(str(friendly_path), cv2.IMREAD_COLOR)
    enemy = cv2.imread(str(enemy_path), cv2.IMREAD_COLOR)
    if friendly is None or enemy is None:
        raise FileNotFoundError("cannot read friendly/enemy map images")
    if friendly.shape != enemy.shape:
        raise ValueError("friendly/enemy map image sizes do not match")
    merged = cv2.max(friendly, enemy)
    return cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)


def mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    binary = np.asarray(mask, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2) for c in contours if c.shape[0] >= 3]


def route_records(
    allocation: np.ndarray,
    physical: dict,
    crossing_routes: set[tuple[int, int]],
) -> pd.DataFrame:
    A = np.asarray(allocation, dtype=float)
    Q = np.asarray(physical["Q"], dtype=float)
    D = np.asarray(physical["D"], dtype=float)
    base_coords = np.asarray(physical["base_coords"], dtype=float)
    target_coords = np.asarray(physical["target_coords"], dtype=float)
    loads = A.sum(axis=0)
    active = np.argwhere(loads > 1e-7)

    rows: list[dict] = []
    for base_id, target_id in active.tolist():
        by_weapon = A[:, base_id, target_id]
        total = float(by_weapon.sum())
        weapon_items = [
            f"W{k}:{int(round(v))}" for k, v in enumerate(by_weapon.tolist()) if v > 1e-7
        ]
        q_weighted = (
            float(np.sum(by_weapon * Q[:, base_id, target_id]) / total)
            if total > 1e-12 else 0.0
        )
        rows.append(
            {
                "base_id": int(base_id),
                "target_id": int(target_id),
                "total_shots": total,
                "weapon_breakdown": ";".join(weapon_items),
                "distance": float(D[base_id, target_id]),
                "avg_interception_on_route": q_weighted,
                "participates_in_crossing": int((base_id, target_id) in crossing_routes),
                "base_row_y": float(base_coords[base_id, 0]),
                "base_col_x": float(base_coords[base_id, 1]),
                "target_row_y": float(target_coords[target_id, 0]),
                "target_col_x": float(target_coords[target_id, 1]),
            }
        )
    return pd.DataFrame(rows)


def selected_scheme_records(schemes: np.ndarray, physical: dict) -> pd.DataFrame:
    schemes = np.asarray(schemes, dtype=np.int64)
    G = np.asarray(physical["G"], dtype=float)
    U = np.asarray(physical["U"], dtype=float)
    rows = []
    for target_id, scheme_id in enumerate(schemes.tolist()):
        demand = U[scheme_id, :, target_id]
        rows.append(
            {
                "target_id": int(target_id),
                "selected_scheme": int(scheme_id),
                "damage_level": float(G[scheme_id, target_id]),
                "raw_scheme_demand": float(np.sum(demand)),
                "weapon_demand": ";".join(
                    f"W{k}:{float(v):g}" for k, v in enumerate(demand.tolist()) if v > 1e-12
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_dark_map(ax, friendly_path: Path, enemy_path: Path) -> tuple[int, int]:
    friendly_mask = load_land_mask(friendly_path)
    enemy_mask = load_land_mask(enemy_path)
    if friendly_mask.shape != enemy_mask.shape:
        raise ValueError("friendly/enemy masks have different sizes")
    height, width = friendly_mask.shape

    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    for contour in mask_contours(friendly_mask):
        ax.fill(contour[:, 0], contour[:, 1], facecolor=FRIENDLY_COLOR, alpha=0.055, zorder=0)
        ax.plot(contour[:, 0], contour[:, 1], color=FRIENDLY_COLOR, lw=1.15, alpha=0.78, zorder=1)
    for contour in mask_contours(enemy_mask):
        ax.fill(contour[:, 0], contour[:, 1], facecolor=TARGET_COLOR, alpha=0.055, zorder=0)
        ax.plot(contour[:, 0], contour[:, 1], color=TARGET_COLOR, lw=1.15, alpha=0.78, zorder=1)

    ax.set_axisbelow(True)
    major_x = max(width // 20, 40)
    major_y = max(height // 12, 40)
    ax.set_xticks(np.arange(0, width + 1, major_x))
    ax.set_yticks(np.arange(0, height + 1, major_y))
    ax.grid(True, color=DARK_GRID, lw=0.45, alpha=0.28)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_color("#314955")
        spine.set_linewidth(0.8)
    return height, width


def draw_satellite_map(ax, friendly_path: Path, enemy_path: Path) -> tuple[int, int]:
    bg = load_merged_satellite(friendly_path, enemy_path)
    height, width = bg.shape[:2]
    ax.imshow(bg, origin="upper", zorder=0)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return height, width


def plot_solution(
    *,
    allocation: np.ndarray,
    schemes: np.ndarray,
    physical: dict,
    metrics: dict,
    profile: dict,
    friendly_path: Path,
    enemy_path: Path,
    style: str,
    method_label: str,
    scenario_id: str,
    output_stem: Path,
    formats: list[str],
    dpi: int,
    route_alpha: float,
    route_width: float,
    width_by_load: bool,
    highlight_crossings: bool,
    show_metrics: bool,
    label_active: bool,
) -> list[Path]:
    base = np.asarray(physical["base_coords"], dtype=float)
    target = np.asarray(physical["target_coords"], dtype=float)
    A = np.asarray(allocation, dtype=float)
    route_load = A.sum(axis=0)
    active = np.argwhere(route_load > 1e-7)

    crossing_pairs = crossing_pairs_from_allocations(A, base, target)
    crossing_routes: set[tuple[int, int]] = set()
    for a, b in crossing_pairs:
        crossing_routes.add(tuple(a))
        crossing_routes.add(tuple(b))

    if style == "dark":
        fig, ax = plt.subplots(figsize=(14.0, 7.6), facecolor=DARK_BG)
        draw_dark_map(ax, friendly_path, enemy_path)
        text_color = "#edf2f4"
        legend_face = "#172832"
        legend_edge = "#526c79"
    else:
        fig, ax = plt.subplots(figsize=(14.0, 7.6))
        draw_satellite_map(ax, friendly_path, enemy_path)
        text_color = "#172026"
        legend_face = "white"
        legend_edge = "#777777"

    max_load = float(np.max(route_load)) if active.size else 1.0

    for base_id, target_id in active.tolist():
        load = float(route_load[base_id, target_id])
        if width_by_load and max_load > 0:
            lw = route_width * (0.65 + 1.35 * np.sqrt(load / max_load))
        else:
            lw = route_width
        is_cross = (base_id, target_id) in crossing_routes
        color = CROSSING_ROUTE_COLOR if (highlight_crossings and is_cross) else TRAJECTORY_COLOR
        alpha = min(0.92, route_alpha + 0.10) if (highlight_crossings and is_cross) else route_alpha
        ax.plot(
            [base[base_id, 1], target[target_id, 1]],
            [base[base_id, 0], target[target_id, 0]],
            color=color,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2,
        )

    active_bases = np.unique(active[:, 0]) if active.size else np.array([], dtype=int)
    active_targets = np.unique(active[:, 1]) if active.size else np.array([], dtype=int)

    if active_targets.size:
        ax.scatter(
            target[active_targets, 1], target[active_targets, 0],
            s=92, facecolors="none", edgecolors=ACTIVE_TARGET_HALO,
            linewidths=5.0, alpha=0.20, zorder=3,
        )

    ax.scatter(
        base[:, 1], base[:, 0],
        s=22, c=FRIENDLY_COLOR, edgecolors="#f5d5d2",
        linewidths=0.65, marker="o", alpha=0.96, zorder=5,
    )
    ax.scatter(
        target[:, 1], target[:, 0],
        s=22, c=TARGET_COLOR, edgecolors="#d9effc",
        linewidths=0.65, marker="o", alpha=0.96, zorder=5,
    )

    if label_active:
        for i in active_bases.tolist():
            ax.annotate(
                f"B{i}", (base[i, 1], base[i, 0]), xytext=(4, 4),
                textcoords="offset points", fontsize=5.5, color=text_color, alpha=0.9, zorder=6,
            )
        for m in active_targets.tolist():
            ax.annotate(
                f"T{m}", (target[m, 1], target[m, 0]), xytext=(4, -8),
                textcoords="offset points", fontsize=5.5, color=text_color, alpha=0.9, zorder=6,
            )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FRIENDLY_COLOR,
               markeredgecolor="#f5d5d2", markersize=5.5, label="Friendly bases"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TARGET_COLOR,
               markeredgecolor="#d9effc", markersize=5.5, label="Adversary targets"),
        Line2D([0], [0], color=TRAJECTORY_COLOR, linewidth=1.2, alpha=0.75, label="Trajectories"),
    ]
    if highlight_crossings:
        handles.append(Line2D([0], [0], color=CROSSING_ROUTE_COLOR, linewidth=1.4,
                              label="Routes involved in crossings"))
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.84,
        facecolor=legend_face,
        edgecolor=legend_edge,
        fontsize=8,
    )
    for text in legend.get_texts():
        text.set_color(text_color)

    if show_metrics:
        profile_score = float(profile.get("profile_score", np.nan))
        ax.set_title(
            f"{method_label} | {scenario_id} | "
            f"routes={int(metrics['active_route_count'])}, "
            f"TCR={metrics['TCR']:.4f}, "
            f"J={profile_score:.4f}",
            color=text_color,
            fontsize=11,
            pad=8,
        )

    fig.tight_layout(pad=0.25)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = output_stem.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if fmt.lower() == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def parse_formats(text: str) -> list[str]:
    formats = [x.strip().lower().lstrip(".") for x in text.split(",") if x.strip()]
    allowed = {"png", "pdf", "svg"}
    unknown = set(formats) - allowed
    if not formats or unknown:
        raise ValueError(f"--formats must use png,pdf,svg; unknown={sorted(unknown)}")
    return list(dict.fromkeys(formats))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--index", type=int, default=0, help="0-based scenario index in the split")
    parser.add_argument("--mode", choices=("fast", "standard"), default="standard")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tactical-style", default="balanced")

    parser.add_argument("--standard-passes", type=int, default=4)
    parser.add_argument("--rho", type=float, default=0.90)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--route-tau", type=float, default=0.25)
    parser.add_argument("--fast-max-fill-ratio", type=float, default=0.05)

    parser.add_argument("--style", choices=("dark", "satellite"), default="dark")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/visualization/deepwta_maps")
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--route-alpha", type=float, default=0.52)
    parser.add_argument("--route-width", type=float, default=0.85)
    parser.add_argument("--width-by-load", action="store_true")
    parser.add_argument("--highlight-crossings", action="store_true")
    parser.add_argument("--hide-metrics", action="store_true")
    parser.add_argument("--label-active", action="store_true", help="Label active B/T ids; not recommended for XLarge")
    args = parser.parse_args()

    if args.index < 0:
        raise ValueError("--index must be >= 0")
    if not (0.0 < args.route_alpha <= 1.0):
        raise ValueError("--route-alpha must be in (0,1]")
    if args.route_width <= 0.0:
        raise ValueError("--route-width must be > 0")
    formats = parse_formats(args.formats)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest = load_dataset_manifest(args.dataset_manifest)
    dataset = ShardedGraphDataset(manifest.root, args.split, TRAIN_CONFIG.shard_cache_size)
    if args.index >= len(dataset):
        raise IndexError(f"--index={args.index} outside {args.split} split of size {len(dataset)}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_deepwta_model(checkpoint, device)
    model.eval()

    graph = dataset[args.index]
    physical = extract_physical(graph)
    scenario_id = str(
        graph_meta_value(getattr(graph, "scenario_id", None), f"{args.split}_{args.index:06d}")
    )

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

    cfg = build_solver_config(args)
    solve_started = time.perf_counter()
    allocation, schemes, report = solve_deepwta(
        allocation_soft,
        scheme_probs,
        physical["B"], physical["U"], physical["G"], physical["G_th"],
        physical["Q"], physical["X"], physical["base_coords"], physical["target_coords"],
        config=cfg,
        return_report=True,
        flight_times=flight_times,
    )
    solver_ms = 1000.0 * (time.perf_counter() - solve_started)

    metrics = scenario_metrics(allocation, schemes, **physical)
    metrics = attach_profile_reference_metrics(
        metrics,
        B=physical["B"], U=physical["U"], G=physical["G"], G_th=physical["G_th"],
        Q=physical["Q"], X=physical["X"],
    )
    profile = profile_score_breakdown(metrics, args.tactical_style)

    if float(metrics.get("hard_feasible", 0.0)) < 1.0:
        raise RuntimeError(f"DeepWTA-{args.mode} returned a hard-infeasible solution for {scenario_id}")

    friendly_path, enemy_path = infer_map_paths(manifest)
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    method_label = f"DeepWTA-{args.mode.capitalize()}"
    stem_name = f"{scenario_id}_{args.mode}_{args.style}"
    output_stem = output_dir / stem_name

    crossing_pairs = crossing_pairs_from_allocations(
        allocation, physical["base_coords"], physical["target_coords"]
    )
    crossing_routes: set[tuple[int, int]] = set()
    for a, b in crossing_pairs:
        crossing_routes.add(tuple(a))
        crossing_routes.add(tuple(b))

    route_df = route_records(allocation, physical, crossing_routes)
    scheme_df = selected_scheme_records(schemes, physical)
    route_csv = output_dir / f"{stem_name}_routes.csv"
    scheme_csv = output_dir / f"{stem_name}_schemes.csv"
    route_df.to_csv(route_csv, index=False)
    scheme_df.to_csv(scheme_csv, index=False)

    npz_path = output_dir / f"{stem_name}_solution.npz"
    np.savez_compressed(
        npz_path,
        allocation=np.asarray(allocation, dtype=np.int64),
        schemes=np.asarray(schemes, dtype=np.int64),
        base_coords=np.asarray(physical["base_coords"], dtype=np.float32),
        target_coords=np.asarray(physical["target_coords"], dtype=np.float32),
    )

    figure_paths = plot_solution(
        allocation=allocation,
        schemes=schemes,
        physical=physical,
        metrics=metrics,
        profile=profile,
        friendly_path=friendly_path,
        enemy_path=enemy_path,
        style=args.style,
        method_label=method_label,
        scenario_id=scenario_id,
        output_stem=output_stem,
        formats=formats,
        dpi=args.dpi,
        route_alpha=args.route_alpha,
        route_width=args.route_width,
        width_by_load=args.width_by_load,
        highlight_crossings=args.highlight_crossings,
        show_metrics=not args.hide_metrics,
        label_active=args.label_active,
    )

    summary = {
        "scenario_id": scenario_id,
        "dataset_name": manifest.dataset_name,
        "scale": manifest.scale,
        "split": args.split,
        "index": int(args.index),
        "method": method_label,
        "checkpoint": str(args.checkpoint),
        "style": args.style,
        "network_time_ms": float(network_ms),
        "solver_time_ms": float(solver_ms),
        "total_time_ms": float(network_ms + solver_ms),
        "metrics": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in metrics.items()},
        "profile": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in profile.items()},
        "solver_report": {
            k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v))
            for k, v in report.items()
            if isinstance(v, (str, bool, int, float, np.integer, np.floating, np.bool_))
        },
        "outputs": {
            "figures": [str(p) for p in figure_paths],
            "routes_csv": str(route_csv),
            "schemes_csv": str(scheme_csv),
            "solution_npz": str(npz_path),
        },
    }
    summary_path = output_dir / f"{stem_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== DeepWTA solution map ===")
    print(f"scenario     : {scenario_id}")
    print(f"method       : {method_label}")
    print(f"hard feasible: {metrics['hard_feasible']:.0f}")
    print(f"WR           : {metrics['WR']:.4f}")
    print(f"active routes: {int(metrics['active_route_count'])}")
    print(f"TCR          : {metrics['TCR']:.4f}")
    print(f"timeliness   : {metrics['avg_distance']:.4f}")
    print(f"inter. ratio : {metrics['avg_interception']:.4f}")
    print(f"J_profile    : {profile['profile_score']:.4f}")
    print(f"network ms   : {network_ms:.2f}")
    print(f"solver ms    : {solver_ms:.2f}")
    for path in figure_paths:
        print(f"figure       : {path}")
    print(f"routes       : {route_csv}")
    print(f"schemes      : {scheme_csv}")
    print(f"solution     : {npz_path}")
    print(f"summary      : {summary_path}")


if __name__ == "__main__":
    main()
