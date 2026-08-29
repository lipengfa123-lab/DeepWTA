#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/"src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from configs.deepwta_train_config import TRAIN_CONFIG
from deepwta.benchmark.adapters import instance_from_graph
from deepwta.data.manifest import load_dataset_manifest
from deepwta.data.sharded_dataset import ShardedGraphDataset
from scripts.reviewer2.common import group_id, save_input_npz, sha256_file


def main():
    ap = argparse.ArgumentParser(description="Export full WTA inputs for Reviewer #2 release.")
    ap.add_argument("--dataset-manifest", type=Path, required=True)
    ap.add_argument("--split", choices=("train","val","test"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = load_dataset_manifest(args.dataset_manifest)
    dataset = ShardedGraphDataset(manifest.root, args.split, TRAIN_CONFIG.shard_cache_size)
    count = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    case_dir = args.output_dir / "cases"
    rows = []
    for i in tqdm(range(count), desc=f"export-inputs:{manifest.dataset_name}:{args.split}"):
        instance = instance_from_graph(dataset[i], dataset_version="v5")
        p = save_input_npz(case_dir, instance, index=i, dataset_name=manifest.dataset_name, split=args.split)
        rows.append({
            "scenario_index": i,
            "scenario_id": instance.instance_id,
            "group_id": group_id(i),
            "scenario_seed": int(instance.seed),
            "K_weapon_types": int(instance.num_weapons),
            "N_bases": int(instance.num_bases),
            "M_targets": int(instance.num_targets),
            "L_schemes": int(instance.num_schemes),
            "input_npz": str(p.relative_to(args.output_dir)),
            "sha256": sha256_file(p),
        })
    pd.DataFrame(rows).to_csv(args.output_dir/"case_index.csv", index=False)
    meta = {
        "protocol": "reviewer2-input-export-v1",
        "dataset_name": manifest.dataset_name,
        "dataset_manifest": str(args.dataset_manifest.resolve()),
        "dataset_root": str(manifest.root),
        "split": args.split,
        "case_count": count,
        "group_definition": "A/B/C = scenario_index modulo 3",
        "stored_inputs": [
            "base_coords[N,2]", "target_coords[M,2]", "B_inventory[K,N]",
            "U_scheme_demand[L,K,M]", "G_damage_level[L,M]",
            "G_th_damage_threshold[M]", "Q_intercept_prob[K,N,M]",
            "D_distance[N,M]", "X_reachable[K,N,M]",
        ],
    }
    (args.output_dir/"dataset_meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print(pd.DataFrame(rows).groupby("group_id").size())
    print(args.output_dir)


if __name__ == "__main__":
    main()
