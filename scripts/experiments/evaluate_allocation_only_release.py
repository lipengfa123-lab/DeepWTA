#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
from deepwta.baselines.learning.e2e_common import make_allocation_env, mark_dead_end
from deepwta.baselines.learning.multi_head_dqn import DQNStateEncoder, MultiHeadDQN
from deepwta.baselines.learning.pointer_net_ac import PointerNetActor
from deepwta.data.sharded_dataset import ShardedGraphDataset
from scripts.experiments.common import group_id, save_solution_npz, scalar_metrics, sha256_file


def sync(d):
    if d.type=="cuda": torch.cuda.synchronize(d)


def load_cache(path,split):
    rows={}
    with (path/f"{split}_fixed_schemes.jsonl").open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r=json.loads(line); rows[int(r["index"])]=r
    return rows


def freeze(m):
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def build_dqn(ckpt,d):
    n=ckpt["network"]
    enc=DQNStateEncoder(int(n["num_weapons"]),max_bases=int(n["max_bases"]),feature_version=int(n.get("feature_version",2)))
    model=MultiHeadDQN(int(n["num_weapons"]),int(n["global_dim"]),candidate_dim=int(n["candidate_dim"]),
        hidden_dim_1=int(n["hidden_dim_1"]),hidden_dim_2=int(n["hidden_dim_2"])).to(d)
    model.load_state_dict(ckpt["model_state"]); return freeze(model),enc


def build_pointer(ckpt,d):
    n=ckpt["network"]; actor=PointerNetActor(int(n["hidden_dim"]),feature_version=int(n.get("feature_version",2))).to(d)
    actor.load_state_dict(ckpt["actor_state"]); return freeze(actor)


def run_dqn(instance,selected,built,d):
    model,enc=built
    env=make_allocation_env(instance,selected,crossing_penalty=0.0,compute_crossings=False,
        emit_step_observation=False,reward_mode="canonical_delta",expose_topology_features=(enc.feature_version>=2),infeasible_penalty=10.0)
    env.reset(); decisions=0; sync(d); t=time.perf_counter()
    with torch.inference_mode():
        while not env.done:
            s=enc.encode(env)
            if not np.any(s.action_mask):
                mark_dead_end(env,"allocation-only DQN dead end"); break
            g=torch.from_numpy(s.global_features).to(d)[None]; c=torch.from_numpy(s.candidate_features).to(d)[None]
            w=torch.tensor([s.weapon_index],dtype=torch.long,device=d); m=torch.from_numpy(s.action_mask).to(d)[None]
            a=int(model.greedy_action(g,c,w,m).item()); env.step(a); decisions+=1
    sync(d); return env.solution(method="MultiHead-DQN-AllocationOnly"),decisions,bool(env._truncated),1000*(time.perf_counter()-t)


def run_pointer(instance,selected,actor,d):
    env=make_allocation_env(instance,selected,crossing_penalty=0.0,compute_crossings=False,
        emit_step_observation=False,expose_topology_features=(getattr(actor,"feature_version",1)>=2))
    env.reset(); sync(d); t=time.perf_counter()
    with torch.inference_mode():
        ro=actor.rollout(env,decode_type="greedy",terminal_cost_fn=None)
    sync(d); return ro.solution,int(ro.num_decisions),bool(ro.truncated),1000*(time.perf_counter()-t)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",choices=("dqn","pointer"),required=True)
    ap.add_argument("--dataset-dir",type=Path,required=True)
    ap.add_argument("--scheme-cache-dir",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--split",choices=("val","test"),required=True)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--warmup",type=int,default=5)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()

    d=torch.device(args.device); ckpt=torch.load(args.checkpoint,map_location=d,weights_only=False)
    if args.method=="dqn": built=build_dqn(ckpt,d); runner=run_dqn; label="MultiHead-DQN-AllocationOnly"
    else: built=build_pointer(ckpt,d); runner=run_pointer; label="PointerNet-AC-AllocationOnly"
    ds=ShardedGraphDataset(args.dataset_dir,args.split,TRAIN_CONFIG.shard_cache_size); cache=load_cache(args.scheme_cache_dir,args.split)
    count=len(ds) if args.limit<=0 else min(args.limit,len(ds))
    for i in range(min(args.warmup,count)):
        inst=instance_from_graph(ds[i],dataset_version="v5"); sel=np.asarray(cache[i]["selected_scheme"],dtype=np.int64)
        runner(inst,sel,built,d)

    args.output_dir.mkdir(parents=True,exist_ok=True); sol_dir=args.output_dir/"solutions"; rows=[]
    for i in tqdm(range(count),desc=f"release-{label}:{args.split}"):
        inst=instance_from_graph(ds[i],dataset_version="v5"); sel=np.asarray(cache[i]["selected_scheme"],dtype=np.int64)
        solution,decisions,truncated,runtime=runner(inst,sel,built,d)
        raw=evaluate_wta_solution(inst,solution,tactical_style="balanced")
        final_solution=canonicalize_nonredundant_solution(inst,solution)
        final=evaluate_wta_solution(inst,final_solution,tactical_style="balanced")
        save_solution_npz(sol_dir,method=label,scenario_index=i,scenario_id=inst.instance_id,
            selected_scheme=final_solution.selected_scheme,allocation=final_solution.allocation,
            selected_scheme_raw=solution.selected_scheme,allocation_raw=solution.allocation)
        rows.append({
            "scenario_index":i,"scenario_id":inst.instance_id,"group_id":group_id(i),"scenario_seed":int(inst.seed),
            "N":inst.num_bases,"M":inst.num_targets,"method":label,
            **scalar_metrics(final),"raw_hard_feasible":float(raw["hard_feasible"]),"raw_profile_score":float(raw["profile_score"]),
            "runtime_ms":runtime,"total_runtime_ms":runtime,"runtime_scope":"allocation_policy_only",
            "allocation_decisions":decisions,"truncated":float(truncated),
        })
    frame=pd.DataFrame(rows); frame.to_csv(args.output_dir/f"{args.split}_details.csv",index=False)
    (args.output_dir/"protocol.json").write_text(json.dumps({
        "protocol":"deepwta-allocation-only-v1","method":label,"checkpoint":str(args.checkpoint.resolve()),
        "checkpoint_sha256":sha256_file(args.checkpoint),"scheme_cache":str(args.scheme_cache_dir.resolve()),
        "split":args.split,"count":count,"inference_only":True,"runtime_excludes_scheme_generation":True
    },indent=2),encoding="utf-8")
    print(frame[["hard_feasible","profile_score","TCR","WR","runtime_ms"]].mean(numeric_only=True))


if __name__=="__main__": main()
