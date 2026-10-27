#!/usr/bin/env python3
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

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/"src"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))

from configs.deepwta_train_config import TRAIN_CONFIG
from deepwta.benchmark.adapters import instance_from_graph
from deepwta.benchmark.canonicalize import canonicalize_nonredundant_solution
from deepwta.benchmark.metrics import evaluate_wta_solution
from deepwta.data.sharded_dataset import ShardedGraphDataset
from scripts.experiments.common import group_id, load_module, save_solution_npz, scalar_metrics, sha256_file


def sync(device):
    if device.type=="cuda": torch.cuda.synchronize(device)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",choices=("dqn","pointer"),required=True)
    ap.add_argument("--dataset-dir",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--split",choices=("val","test"),required=True)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--warmup-count",type=int,default=10)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()

    official=load_module(ROOT/"scripts/evaluate/evaluate_e2e_learning_baselines.py","r2_official_e2e")
    device=torch.device(args.device)
    ckpt=official.torch_load_cross_platform(args.checkpoint,map_location=device)
    if args.method=="dqn":
        built=official.build_dqn(ckpt,device); runner=official.run_dqn; label="MultiHead-DQN-E2E"
    else:
        built=official.build_pointer(ckpt,device); runner=official.run_pointer; label="PointerNet-AC-E2E"
    ds=ShardedGraphDataset(args.dataset_dir,args.split,TRAIN_CONFIG.shard_cache_size)
    count=len(ds) if args.limit<=0 else min(args.limit,len(ds))
    for i in range(min(args.warmup_count,count)):
        instance=instance_from_graph(ds[i],dataset_version="v5")
        runner(instance,built,device)
    sync(device)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    sol_dir=args.output_dir/"solutions"; rows=[]
    for i in tqdm(range(count),desc=f"release-{label}:{args.split}"):
        instance=instance_from_graph(ds[i],dataset_version="v5")
        solution,scheme_ms,allocation_ms,sd,ad,truncated=runner(instance,built,device)
        raw=evaluate_wta_solution(instance,solution,tactical_style="balanced")
        final_solution=canonicalize_nonredundant_solution(instance,solution)
        final=evaluate_wta_solution(instance,final_solution,tactical_style="balanced")
        save_solution_npz(sol_dir,method=label,scenario_index=i,scenario_id=instance.instance_id,
            selected_scheme=final_solution.selected_scheme,allocation=final_solution.allocation,
            selected_scheme_raw=solution.selected_scheme,allocation_raw=solution.allocation)
        rows.append({
            "scenario_index":i,"scenario_id":instance.instance_id,"group_id":group_id(i),"scenario_seed":int(instance.seed),
            "N":instance.num_bases,"M":instance.num_targets,"method":label,"method_seed":args.seed,
            **scalar_metrics(final),
            "raw_hard_feasible":float(raw.get("hard_feasible",np.nan)),
            "raw_profile_score":float(raw.get("profile_score",np.nan)),
            "scheme_selection_ms":scheme_ms,"allocation_ms":allocation_ms,
            "total_runtime_ms":scheme_ms+allocation_ms,"runtime_ms":scheme_ms+allocation_ms,
            "runtime_scope":"scheme_policy+allocation_policy",
            "scheme_decisions":sd,"allocation_decisions":ad,"truncated":float(truncated),
        })
    frame=pd.DataFrame(rows); frame.to_csv(args.output_dir/f"{args.split}_details.csv",index=False)
    (args.output_dir/"protocol.json").write_text(json.dumps({
        "protocol":"deepwta-e2e-learning-v1","method":label,"checkpoint":str(args.checkpoint.resolve()),
        "checkpoint_sha256":sha256_file(args.checkpoint),"split":args.split,"count":count,
        "inference_only":True,"optimizer_loaded":False,"decision_variables":f"solutions/{label}/*.npz"
    },indent=2),encoding="utf-8")
    print(frame[["hard_feasible","profile_score","TCR","WR","total_runtime_ms"]].mean(numeric_only=True))
    print(args.output_dir)


if __name__=="__main__": main()
