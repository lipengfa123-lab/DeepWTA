#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/"src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from deepwta.data.manifest import load_dataset_manifest
from deepwta.evaluation.exact_metrics import scenario_metrics
from deepwta.evaluation.profile_objective import (
    attach_profile_reference_metrics,
    profile_score_breakdown,
)


def load_viz_module():
    path = ROOT/"scripts/visualization/plot_deepwta_solution_map.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("deepwta_saved_solution_viz_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def physical_from_npz(z):
    return {
        "B": np.asarray(z["B_inventory"]),
        "U": np.asarray(z["U_scheme_demand"]),
        "G": np.asarray(z["G_damage_level"]),
        "G_th": np.asarray(z["G_th_damage_threshold"]),
        "Q": np.asarray(z["Q_intercept_prob"]),
        "D": np.asarray(z["D_distance"]),
        "X": np.asarray(z["X_reachable"]),
        "base_coords": np.asarray(z["base_coords"]),
        "target_coords": np.asarray(z["target_coords"]),
    }


def scalar(v):
    if isinstance(v, (int,float,np.integer,np.floating)):
        return float(v)
    return v


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-manifest",type=Path,required=True)
    ap.add_argument("--input-npz",type=Path,required=True)
    ap.add_argument("--solution-npz",type=Path,required=True)
    ap.add_argument("--method-label",required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--style",choices=("dark","satellite"),default="dark")
    ap.add_argument("--formats",default="png,pdf")
    ap.add_argument("--dpi",type=int,default=320)
    ap.add_argument("--route-alpha",type=float,default=0.52)
    ap.add_argument("--route-width",type=float,default=0.85)
    ap.add_argument("--width-by-load",action="store_true")
    ap.add_argument("--highlight-crossings",action="store_true")
    ap.add_argument("--hide-metrics",action="store_true")
    ap.add_argument("--label-active",action="store_true")
    ap.add_argument("--allow-infeasible",action="store_true")
    args=ap.parse_args()

    viz=load_viz_module()
    formats=viz.parse_formats(args.formats)
    manifest=load_dataset_manifest(args.dataset_manifest, require_root=False)

    with np.load(args.input_npz,allow_pickle=False) as z:
        physical=physical_from_npz(z)
        scenario_id=str(z["scenario_id"].item())
        scenario_index=int(z["scenario_index"].item())

    with np.load(args.solution_npz,allow_pickle=False) as z:
        allocation=np.asarray(z["allocation"],dtype=np.int64)
        schemes=np.asarray(z["selected_scheme"],dtype=np.int64)
        saved_sid=str(z["scenario_id"].item()) if "scenario_id" in z.files else scenario_id

    if saved_sid != scenario_id:
        raise RuntimeError(f"scenario mismatch: input={scenario_id}, solution={saved_sid}")

    metrics=scenario_metrics(allocation,schemes,**physical)
    metrics=attach_profile_reference_metrics(
        metrics,
        B=physical["B"],U=physical["U"],G=physical["G"],G_th=physical["G_th"],
        Q=physical["Q"],X=physical["X"],
    )
    profile=profile_score_breakdown(metrics,"balanced")
    if float(metrics.get("hard_feasible",0.0)) < 1.0 and not args.allow_infeasible:
        raise RuntimeError(f"{args.method_label} solution is hard-infeasible on {scenario_id}")

    friendly_path,enemy_path=viz.infer_map_paths(manifest)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    safe_method=args.method_label.replace("/","_").replace(" ","_")
    stem=args.output_dir/f"{scenario_index:06d}_{safe_method}_{args.style}"

    crossing_pairs=viz.crossing_pairs_from_allocations(
        allocation,physical["base_coords"],physical["target_coords"]
    )
    crossing_routes=set()
    for a,b in crossing_pairs:
        crossing_routes.add(tuple(a)); crossing_routes.add(tuple(b))

    routes=viz.route_records(allocation,physical,crossing_routes)
    schemes_df=viz.selected_scheme_records(schemes,physical)
    routes.to_csv(stem.with_name(stem.name+"_routes.csv"),index=False)
    schemes_df.to_csv(stem.with_name(stem.name+"_schemes.csv"),index=False)

    figs=viz.plot_solution(
        allocation=allocation,
        schemes=schemes,
        physical=physical,
        metrics=metrics,
        profile=profile,
        friendly_path=friendly_path,
        enemy_path=enemy_path,
        style=args.style,
        method_label=args.method_label,
        scenario_id=scenario_id,
        output_stem=stem,
        formats=formats,
        dpi=args.dpi,
        route_alpha=args.route_alpha,
        route_width=args.route_width,
        width_by_load=args.width_by_load,
        highlight_crossings=args.highlight_crossings,
        show_metrics=not args.hide_metrics,
        label_active=args.label_active,
    )

    summary={
        "protocol":"saved-deepwta-solution-visualization-v1",
        "scenario_index":scenario_index,
        "scenario_id":scenario_id,
        "method":args.method_label,
        "source_input_npz":str(args.input_npz.resolve()),
        "source_solution_npz":str(args.solution_npz.resolve()),
        "reran_model":False,
        "reran_solver":False,
        "metrics":{k:scalar(v) for k,v in metrics.items()},
        "profile":{k:scalar(v) for k,v in profile.items()},
        "figure_files":[str(p) for p in figs],
    }
    summary_path=stem.with_name(stem.name+"_summary.json")
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")

    print(f"scenario : {scenario_id}")
    print(f"method   : {args.method_label}")
    print(f"feasible : {metrics['hard_feasible']:.0f}")
    print(f"J        : {profile['profile_score']:.6f}")
    print(f"TCR      : {metrics['TCR']:.6f}")
    print(f"WR       : {metrics['WR']:.6f}")
    for p in figs:
        print(f"figure   : {p}")


if __name__=="__main__":
    main()
