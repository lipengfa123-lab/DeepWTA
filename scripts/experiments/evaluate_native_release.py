#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/"src"):
    if str(p) not in sys.path:
        sys.path.insert(0,str(p))

from configs.deepwta_train_config import TRAIN_CONFIG
from deepwta.data.manifest import load_dataset_manifest
from deepwta.data.sharded_dataset import ShardedGraphDataset
from deepwta.benchmark.canonicalize import canonicalize_nonredundant_allocation
from deepwta.evaluation.exact_metrics import scenario_metrics
from deepwta.evaluation.profile_objective import attach_profile_reference_metrics, profile_score_breakdown
from deepwta.utils.numeric import extract_physical, graph_meta_value
from scripts.experiments.common import group_id, load_module, save_solution_npz, scalar_metrics


DISPLAY={"fra_dc":"FRA-DC","sa":"SA-MC","ga":"GA-MC"}
OFFSET={"fra_dc":7,"sa":17,"ga":31}


def parse_ints(text):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_methods(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def build_budgets(mod, base_profile: str, fra_max: int, fra_stall: int):
    try:
        budgets = copy.deepcopy(mod.baseline_budgets(base_profile))
    except Exception as e:
        raise RuntimeError(
            f"existing baseline profile {base_profile!r} cannot be loaded. "
            "Use a profile that already exists in configs/native_baseline_config.py."
        ) from e

    if "fra_dc" not in budgets:
        raise RuntimeError(
            f"'fra_dc' is absent from existing profile {base_profile!r}; "
            "the FRA-DC migration is not active in the current source tree."
        )

    base_fra = copy.deepcopy(budgets["fra_dc"])
    base_fra["max_iterations"] = int(fra_max)
    base_fra["stall_iterations"] = int(fra_stall)
    budgets["fra_dc"] = base_fra
    return budgets


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-manifest",type=Path,required=True)
    ap.add_argument("--split",choices=("val","test"),required=True)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--methods",default="fra_dc,sa,ga")
    ap.add_argument("--seeds",default="2026,2027,2028")
    ap.add_argument("--base-budget-profile",default="pilot")
    ap.add_argument("--fra-dc-max-iterations",type=int,default=250)
    ap.add_argument("--fra-dc-stall-iterations",type=int,default=200)
    ap.add_argument("--tactical-style",default="balanced")
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()

    mod=load_module(ROOT/"scripts/evaluate/evaluate_native_multichoice_baselines.py","r2_native_official")
    methods=parse_methods(args.methods)
    seeds=parse_ints(args.seeds)
    unknown=set(methods)-set(DISPLAY)
    if unknown:
        raise ValueError(f"unknown methods {unknown}")

    budgets=build_budgets(
        mod,
        args.base_budget_profile,
        args.fra_dc_max_iterations,
        args.fra_dc_stall_iterations,
    )

    print("release native budget protocol:")
    print(f"  base profile : {args.base_budget_profile}")
    for m in methods:
        print(f"  {DISPLAY[m]:8s}: {budgets[m]}")
    print(f"  seeds        : {seeds}")

    manifest=load_dataset_manifest(args.dataset_manifest)
    ds=ShardedGraphDataset(manifest.root,args.split,TRAIN_CONFIG.shard_cache_size)
    count=len(ds) if args.limit<=0 else min(args.limit,len(ds))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    sol_dir=args.output_dir/"solutions"
    rows=[]

    for i in tqdm(range(count),desc=f"release-native:{args.split}"):
        graph=ds[i]
        physical=extract_physical(graph)
        sid=str(graph_meta_value(getattr(graph,"scenario_id",None),f"{args.split}_{i:06d}"))
        scen_seed=int(graph_meta_value(getattr(graph,"scenario_seed",None),-1))
        N=int(physical["Q"].shape[1])
        M=int(physical["Q"].shape[2])

        for method in methods:
            for seed in seeds:
                problem=mod.make_problem(physical,args.tactical_style)
                solver_seed=seed+i*100003+OFFSET[method]
                result=mod.run_method(problem,method,solver_seed,budgets[method])

                canonical=canonicalize_nonredundant_allocation(
                    result.allocation,result.scheme_idx,physical["U"],physical["Q"]
                )
                metrics=scenario_metrics(canonical,result.scheme_idx,**physical)
                metrics=attach_profile_reference_metrics(
                    metrics,
                    B=physical["B"],U=physical["U"],G=physical["G"],
                    G_th=physical["G_th"],Q=physical["Q"],X=physical["X"],
                )
                profile=profile_score_breakdown(metrics,args.tactical_style)

                label=DISPLAY[method]
                method_with_seed=f"{label}__seed{seed}"
                save_solution_npz(
                    sol_dir,
                    method=method_with_seed,
                    scenario_index=i,
                    scenario_id=sid,
                    selected_scheme=result.scheme_idx,
                    allocation=np.rint(canonical).astype(np.int64),
                    selected_scheme_raw=result.scheme_idx,
                    allocation_raw=np.rint(result.allocation).astype(np.int64),
                    extra={
                        "algorithm_seed":seed,
                        "solver_seed":solver_seed,
                        "base_budget_profile":args.base_budget_profile,
                        "effective_budget":budgets[method],
                    },
                )

                row={
                    "scenario_index":i,
                    "scenario_id":sid,
                    "group_id":group_id(i),
                    "scenario_seed":scen_seed,
                    "N":N,
                    "M":M,
                    "method":label,
                    "method_seed":seed,
                    "solver_seed":solver_seed,
                    "base_budget_profile":args.base_budget_profile,
                    "fra_dc_max_iterations_override":args.fra_dc_max_iterations if method=="fra_dc" else np.nan,
                    "fra_dc_stall_iterations_override":args.fra_dc_stall_iterations if method=="fra_dc" else np.nan,
                    **scalar_metrics(metrics),
                    **scalar_metrics(profile),
                    "runtime_ms":1000*float(result.runtime),
                    "total_runtime_ms":1000*float(result.runtime),
                    "runtime_scope":"full_classical_solver",
                    "iterations":int(result.iterations),
                    "best_iteration":int(result.best_iteration),
                    "evaluations":int(result.evaluations),
                    "stop_reason":str(result.stop_reason),
                }
                for k,v in result.diagnostics.items():
                    try:
                        row["diag_"+str(k)]=float(v)
                    except Exception:
                        pass
                rows.append(row)

    frame=pd.DataFrame(rows)
    frame.to_csv(args.output_dir/f"{args.split}_details.csv",index=False)
    protocol={
        "protocol":"deepwta-native-three-seed-v2",
        "dataset_manifest":str(args.dataset_manifest.resolve()),
        "split":args.split,
        "count":count,
        "methods":methods,
        "display_labels":[DISPLAY[m] for m in methods],
        "algorithm_seeds":seeds,
        "base_budget_profile":args.base_budget_profile,
        "fra_dc_local_override":{
            "max_iterations":args.fra_dc_max_iterations,
            "stall_iterations":args.fra_dc_stall_iterations,
        },
        "effective_budgets":{m:budgets[m] for m in methods},
        "global_config_modified":False,
        "decision_variables":"solutions/<method>__seed<seed>/*.npz",
    }
    (args.output_dir/"protocol.json").write_text(
        json.dumps(protocol,indent=2,ensure_ascii=False),encoding="utf-8"
    )

    print(
        frame.groupby(["method","method_seed"])[
            ["hard_feasible","profile_score","TCR","runtime_ms"]
        ].mean()
    )
    print(args.output_dir)


if __name__=="__main__":
    main()
