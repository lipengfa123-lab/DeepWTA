#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.experiments.common import bootstrap_ci, exact_mcnemar_p

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon=None

ORDER=["FRA-DC","DeepWTA-Standard","SA-MC","GA-MC","MultiHead-DQN-E2E","PointerNet-AC-E2E"]


def load(path):
    f=pd.read_csv(path)
    if "scenario_index" not in f: raise ValueError(path)
    return f


def scenario_units(frame):
    key=["method","scenario_index","scenario_id","group_id"]
    numeric=frame.select_dtypes(include=[np.number]).columns.tolist()
    drop={"method_seed","solver_seed","scenario_seed","scenario_index"}
    agg=[c for c in numeric if c not in drop]
    out=frame.groupby(key,as_index=False)[agg].mean()
    return out


def summary(units,boot,seed):
    rows=[]
    for method in ORDER:
        g=units[units.method==method]
        if g.empty: continue
        feas=pd.to_numeric(g.hard_feasible,errors="coerce")
        fg=g[feas>0.5]
        row={"Method":method,"N":len(g),"Feasible":float(feas.mean())}
        for col,name in [
            ("profile_score","J_feasible"),("WR","WR"),
            ("TCR","TCR"),("avg_distance","Timeliness"),("avg_interception","Inter. ratio")
        ]:
            src=fg if col=="profile_score" else g
            vals=pd.to_numeric(src[col],errors="coerce").dropna().to_numpy()
            m,lo,hi=bootstrap_ci(vals,seed=seed+len(rows),samples=boot)
            row[name]=m; row[name+"_CI95_Low"]=lo; row[name+"_CI95_High"]=hi
        rt=pd.to_numeric(g.total_runtime_ms if "total_runtime_ms" in g else g.runtime_ms,errors="coerce")
        row["RuntimeMeanMs"]=float(rt.mean()); row["RuntimeMedianMs"]=float(rt.median()); row["RuntimeP95Ms"]=float(rt.quantile(.95))
        rows.append(row)
    return pd.DataFrame(rows)


def paired(units,reference,boot,seed):
    ref=units[units.method==reference][["scenario_index","hard_feasible","profile_score"]].rename(
        columns={"hard_feasible":"ref_feas","profile_score":"ref_j"})
    rows=[]
    for method in ORDER:
        if method==reference: continue
        b=units[units.method==method][["scenario_index","hard_feasible","profile_score"]].rename(
            columns={"hard_feasible":"b_feas","profile_score":"b_j"})
        p=ref.merge(b,on="scenario_index")
        if p.empty: continue
        both=p[(p.ref_feas>0.5)&(p.b_feas>0.5)]
        d=(both.b_j-both.ref_j).to_numpy(float)
        mean,lo,hi=bootstrap_ci(d,seed=seed+len(rows),samples=boot)
        wp=np.nan
        if wilcoxon is not None and len(d) and not np.allclose(d,0):
            try: wp=float(wilcoxon(d).pvalue)
            except Exception: pass
        a=int(((p.ref_feas>0.5)&~(p.b_feas>0.5)).sum())
        bb=int((~(p.ref_feas>0.5)&(p.b_feas>0.5)).sum())
        rows.append({"Baseline":method,"Reference":reference,"PairedN":len(p),"BothFeasibleN":len(both),
            "DeltaJ_BaselineMinusReference":mean,"CI95_Low":lo,"CI95_High":hi,"WilcoxonP":wp,
            "ReferenceFeasible_BaselineInfeasible":a,"ReferenceInfeasible_BaselineFeasible":bb,
            "McNemarExactP":exact_mcnemar_p(a,bb)})
    return pd.DataFrame(rows)


def bar(summary,col,ylabel,path,log=False,ylim=None):
    fig,ax=plt.subplots(figsize=(7.8,4.8))
    x=np.arange(len(summary)); vals=summary[col].to_numpy(float)
    ax.bar(x,vals); ax.set_xticks(x,summary.Method,rotation=18,ha="right"); ax.set_ylabel(ylabel)
    if log: ax.set_yscale("log")
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis="y",alpha=.25); fig.tight_layout()
    fig.savefig(path.with_suffix(".png"),dpi=250,bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--deepwta",type=Path,required=True)
    ap.add_argument("--dqn",type=Path,required=True)
    ap.add_argument("--pointer",type=Path,required=True)
    ap.add_argument("--fra-dc",dest="fra",type=Path,required=True)
    ap.add_argument("--sa",type=Path,required=True)
    ap.add_argument("--ga",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--bootstrap-samples",type=int,default=10000)
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args()
    frames=[]
    for p in [args.deepwta,args.dqn,args.pointer,args.fra,args.sa,args.ga]:
        frames.append(load(p))
    raw=pd.concat(frames,ignore_index=True,sort=False)
    units=scenario_units(raw)
    args.output_dir.mkdir(parents=True,exist_ok=True); figdir=args.output_dir/"figures"; figdir.mkdir(exist_ok=True)
    raw.to_csv(args.output_dir/"all_result_rows.csv",index=False)
    units.to_csv(args.output_dir/"scenario_units.csv",index=False)
    summ=summary(units,args.bootstrap_samples,args.seed); summ.to_csv(args.output_dir/"table_main.csv",index=False)
    paired_deep=paired(units,"DeepWTA-Standard",args.bootstrap_samples,args.seed+100); paired_deep.to_csv(args.output_dir/"paired_vs_deepwta.csv",index=False)
    classical=raw[raw.method.isin(["FRA-DC","SA-MC","GA-MC"])]
    seed_summary=classical.groupby(["method","method_seed"],as_index=False).agg(
        hard_feasible=("hard_feasible","mean"),profile_score=("profile_score","mean"),
        TCR=("TCR","mean"),WR=("WR","mean"),
        runtime_ms=("runtime_ms","mean"))
    seed_summary.to_csv(args.output_dir/"classical_three_seed_summary.csv",index=False)
    group_summary=units.groupby(["group_id","method"],as_index=False).agg(
        hard_feasible=("hard_feasible","mean"),profile_score=("profile_score","mean"),
        TCR=("TCR","mean"),runtime_ms=("total_runtime_ms","mean"))
    group_summary.to_csv(args.output_dir/"three_group_summary.csv",index=False)
    bar(summ,"Feasible","Hard-feasibility rate",figdir/"feasibility",ylim=(0,1.05))
    bar(summ,"J_feasible","Canonical profile J (feasible cases)",figdir/"objective_J")
    bar(summ,"TCR","Trajectory-crossing rate",figdir/"TCR")
    bar(summ,"WR","Weapon consumption ratio to raw lower bound",figdir/"WR")
    bar(summ,"RuntimeMedianMs","Median online runtime (ms)",figdir/"runtime",log=True)
    fig,ax=plt.subplots(figsize=(7.0,5.0))
    for _,r in summ.iterrows():
        ax.scatter(r.RuntimeMedianMs,r.J_feasible,s=80)
        ax.annotate(r.Method,(r.RuntimeMedianMs,r.J_feasible),xytext=(5,5),textcoords="offset points",fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Median online runtime (ms, log scale)"); ax.set_ylabel("J (lower is better)")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(figdir/"quality_runtime_pareto.png",dpi=250,bbox_inches="tight"); fig.savefig(figdir/"quality_runtime_pareto.pdf",bbox_inches="tight"); plt.close(fig)
    report=[
        "# release Main Result Report","",
        "All stochastic classical methods are first averaged across the three algorithm seeds within each WTA case.",
        "Statistical tests therefore use WTA cases, not seed-runs, as the experimental unit.","",
        "## Main table","",summ.to_markdown(index=False,floatfmt=".5g"),"",
        "## Paired vs DeepWTA-Standard","",paired_deep.to_markdown(index=False,floatfmt=".5g"),"",
    ]
    (args.output_dir/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print(summ.to_string(index=False)); print(args.output_dir)


if __name__=="__main__": main()
