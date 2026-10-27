#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.experiments.common import bootstrap_ci


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--deepwta-details",type=Path,required=True)
    ap.add_argument("--deepwta-solutions",type=Path,required=True)
    ap.add_argument("--dqn-details",type=Path,required=True)
    ap.add_argument("--pointer-details",type=Path,required=True)
    ap.add_argument("--scheme-cache-dir",type=Path,default=None,
        help="optional: directory with <split>_fixed_schemes.jsonl for the scheme-match audit")
    ap.add_argument("--split",choices=("val","test"),required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    deep=pd.read_csv(args.deepwta_details)
    dqn=pd.read_csv(args.dqn_details); ptr=pd.read_csv(args.pointer_details)
    frames=[deep,dqn,ptr]; allf=pd.concat(frames,ignore_index=True,sort=False)
    cache={}
    cache_file = (args.scheme_cache_dir/f"{args.split}_fixed_schemes.jsonl") if args.scheme_cache_dir else None
    if cache_file is not None and cache_file.is_file():
        with cache_file.open("r",encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r=json.loads(line); cache[int(r["index"])]=r
    audits=[]
    for i in sorted(dqn.scenario_index.astype(int).unique()):
        p=args.deepwta_solutions/"DeepWTA-Standard"/f"idx_{i:06d}.npz"
        if not p.is_file():
            audits.append({"scenario_index":i,"scheme_match":0,"reason":"missing_deepwta_solution"}); continue
        z=np.load(p,allow_pickle=False); a=np.asarray(z["selected_scheme"],dtype=np.int64)
        if i not in cache:
            audits.append({"scenario_index":i,"scheme_match":np.nan,"reason":"scheme_cache_unavailable"}); continue
        b=np.asarray(cache[i]["selected_scheme"],dtype=np.int64)
        audits.append({"scenario_index":i,"scenario_id":str(cache[i].get("scenario_id","")),
            "scheme_match":float(a.shape==b.shape and np.array_equal(a,b))})
    audit=pd.DataFrame(audits)
    known = audit[audit["reason"].isna()] if "reason" in audit.columns else audit
    match=float(known.scheme_match.mean()) if len(known) else float("nan")
    args.output_dir.mkdir(parents=True,exist_ok=True); audit.to_csv(args.output_dir/"scheme_match_audit.csv",index=False)
    allf.to_csv(args.output_dir/"comparison_rows.csv",index=False)
    rows=[]
    order=["DeepWTA-Standard","MultiHead-DQN-AllocationOnly","PointerNet-AC-AllocationOnly"]
    for m in order:
        g=allf[allf.method==m]; feas=g.hard_feasible.astype(float)
        fg=g[feas>0.5]
        rt=pd.to_numeric(g.total_runtime_ms if "total_runtime_ms" in g else g.runtime_ms,errors="coerce")
        rows.append({"Method":m,"N":len(g),"Feasible":feas.mean(),"J":fg.profile_score.mean() if len(fg) else np.nan,
            "WR":g.WR.mean(),"TCR":g.TCR.mean(),
            "Timeliness":g.avg_distance.mean(),"Inter. ratio":g.avg_interception.mean(),
            "RuntimeMedianMs":rt.median(),"RuntimeP95Ms":rt.quantile(.95)})
    table=pd.DataFrame(rows); table.to_csv(args.output_dir/"table_allocation_only.csv",index=False)
    ref=deep[["scenario_index","hard_feasible","profile_score"]].rename(columns={"profile_score":"ref_j","hard_feasible":"ref_f"})
    pairs=[]
    for m,f in [("MultiHead-DQN-AllocationOnly",dqn),("PointerNet-AC-AllocationOnly",ptr)]:
        p=ref.merge(f[["scenario_index","hard_feasible","profile_score"]],on="scenario_index")
        p=p[(p.ref_f>0.5)&(p.hard_feasible>0.5)]
        d=(p.profile_score-p.ref_j).to_numpy(float)
        mean,lo,hi=bootstrap_ci(d,samples=10000)
        pairs.append({"Baseline":m,"BothFeasibleN":len(p),"DeltaJ_BaselineMinusDeepWTA":mean,"CI95_Low":lo,"CI95_High":hi})
    pd.DataFrame(pairs).to_csv(args.output_dir/"paired_statistics.csv",index=False)
    figdir=args.output_dir/"figures"; figdir.mkdir(exist_ok=True)
    for col,y in [("J","Canonical profile J"),("TCR","TCR"),("WR","WR"),("Feasible","Feasibility")]:
        fig,ax=plt.subplots(figsize=(6.2,4.3)); ax.bar(np.arange(3),table[col])
        ax.set_xticks(np.arange(3),table.Method,rotation=15,ha="right"); ax.set_ylabel(y); ax.grid(axis="y",alpha=.25); fig.tight_layout()
        fig.savefig(figdir/f"{col.replace('/','_')}.png",dpi=250,bbox_inches="tight"); fig.savefig(figdir/f"{col.replace('/','_')}.pdf",bbox_inches="tight"); plt.close(fig)
    (args.output_dir/"AUDIT.json").write_text(json.dumps({
        "scheme_match_rate": None if math.isnan(match) else match,
        "scheme_match_audit": "skipped (scheme cache not redistributed)" if math.isnan(match) else "completed",
        "valid_controlled_quality_comparison": bool(match==1.0),
        "runtime_semantics":{
            "DeepWTA-Standard":"network + standard solver",
            "DQN/Pointer AllocationOnly":"allocation policy only; frozen scheme generation excluded"
        }
    },indent=2),encoding="utf-8")
    print(table.to_string(index=False))
    if math.isnan(match):
        print("scheme_match_rate=skipped (scheme cache not provided)")
    else:
        print(f"scheme_match_rate={match:.6f}")
        if match < 1.0: raise RuntimeError("frozen scheme cache != evaluated DeepWTA selected schemes")


if __name__=="__main__": main()
