#!/usr/bin/env python3
"""Reviewer-2 DeepWTA-Standard evaluator with full decision-variable dumps."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/"src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from configs.deepwta_solver_config import DeepWTASolverConfig
from configs.deepwta_train_config import TRAIN_CONFIG
from deepwta.data.manifest import load_dataset_manifest
from deepwta.data.sharded_dataset import ShardedGraphDataset
from deepwta.evaluation.exact_metrics import scenario_metrics
from deepwta.evaluation.profile_objective import attach_profile_reference_metrics, profile_score_breakdown
from deepwta.solver import solve_deepwta
from deepwta.utils.checkpoint_io import build_deepwta_model
from deepwta.utils.numeric import extract_physical, graph_meta_value
from scripts.reviewer2.common import group_id, save_solution_npz, scalar_metrics, sha256_file


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reconstruct_allocation(graph, edge_values):
    x=graph.x.detach().cpu(); edge_index=graph.edge_index.detach().cpu()
    base_nodes=torch.nonzero(x[:,2]<0.5,as_tuple=False).reshape(-1)
    target_nodes=torch.nonzero(x[:,2]>=0.5,as_tuple=False).reshape(-1)
    base_lookup=torch.full((x.size(0),),-1,dtype=torch.long)
    target_lookup=torch.full((x.size(0),),-1,dtype=torch.long)
    base_lookup[base_nodes]=torch.arange(base_nodes.numel())
    target_lookup[target_nodes]=torch.arange(target_nodes.numel())
    b=base_lookup[edge_index[0]]; m=target_lookup[edge_index[1]]
    values=edge_values.detach().float().cpu()
    A=torch.zeros((int(values.size(1)),int(base_nodes.numel()),int(target_nodes.numel())),dtype=values.dtype)
    A[:,b,m]=values.T
    return A.numpy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--dataset-manifest",type=Path,required=True)
    ap.add_argument("--split",choices=("val","test"),required=True)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--warmup",type=int,default=3)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--tactical-style",default="balanced")
    ap.add_argument("--standard-passes",type=int,default=4)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()

    device=torch.device(args.device)
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=build_deepwta_model(ckpt,device)
    manifest=load_dataset_manifest(args.dataset_manifest)
    ds=ShardedGraphDataset(manifest.root,args.split,TRAIN_CONFIG.shard_cache_size)
    count=len(ds) if args.limit<=0 else min(args.limit,len(ds))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    sol_dir=args.output_dir/"solutions"

    if count and args.warmup:
        g=ds[0].to(device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(g,scheme_mode="hard",scheme_tau=1.0,gumbel_hard=True)
        sync(device); del g

    cfg=DeepWTASolverConfig(max_passes=args.standard_passes)
    rows=[]
    for i in tqdm(range(count),desc=f"R2-DeepWTA:{args.split}"):
        graph=ds[i]; physical=extract_physical(graph)
        sid=str(graph_meta_value(getattr(graph,"scenario_id",None),f"{args.split}_{i:06d}"))
        gd=graph.to(device)
        sync(device); t0=time.perf_counter()
        with torch.inference_mode():
            out=model(gd,scheme_mode="hard",scheme_tau=1.0,gumbel_hard=True)
        sync(device); network_ms=1000*(time.perf_counter()-t0)
        soft=reconstruct_allocation(gd,out.allocation)
        scheme_probs=out.scheme_probs.detach().float().cpu().numpy()
        flight=physical["D"]/max(float(np.max(physical["D"])),1e-12)
        t1=time.perf_counter()
        allocation, schemes, report=solve_deepwta(
            soft,scheme_probs,physical["B"],physical["U"],physical["G"],physical["G_th"],
            physical["Q"],physical["X"],physical["base_coords"],physical["target_coords"],
            config=cfg,return_report=True,flight_times=flight,
        )
        solver_ms=1000*(time.perf_counter()-t1)
        metrics=scenario_metrics(allocation,schemes,**physical)
        metrics=attach_profile_reference_metrics(metrics,B=physical["B"],U=physical["U"],G=physical["G"],
            G_th=physical["G_th"],Q=physical["Q"],X=physical["X"])
        profile=profile_score_breakdown(metrics,args.tactical_style)
        save_solution_npz(sol_dir,method="DeepWTA-Standard",scenario_index=i,scenario_id=sid,
            selected_scheme=schemes,allocation=np.rint(allocation).astype(np.int64),
            extra={"network_ms":network_ms,"solver_ms":solver_ms})
        rows.append({
            "scenario_index":i,"scenario_id":sid,"group_id":group_id(i),"scenario_seed":int(graph_meta_value(getattr(graph,"scenario_seed",None),-1)),
            "N":int(physical["Q"].shape[1]),"M":int(physical["Q"].shape[2]),
            "method":"DeepWTA-Standard","method_seed":args.seed,
            **scalar_metrics(metrics),**scalar_metrics(profile),
            "network_time_ms":network_ms,"runtime_ms":solver_ms,"total_runtime_ms":network_ms+solver_ms,
            "runtime_scope":"network+standard_solver","construction_fallback_used":float(bool(report.get("construction_fallback_used",False))),
        })
    frame=pd.DataFrame(rows); frame.to_csv(args.output_dir/f"{args.split}_details.csv",index=False)
    protocol={
        "protocol":"deepwta-standard-v1","checkpoint":args.checkpoint.as_posix(),
        "checkpoint_sha256":sha256_file(args.checkpoint),"dataset_manifest":args.dataset_manifest.as_posix(),
        "split":args.split,"count":count,"solver_config":cfg.as_dict(),"decision_variables":"solutions/DeepWTA-Standard/*.npz"
    }
    (args.output_dir/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    print(frame[["hard_feasible","profile_score","crossing_pair_rate","ammo_ratio_to_raw_lb","total_runtime_ms"]].mean(numeric_only=True))
    print(args.output_dir)


if __name__=="__main__": main()
