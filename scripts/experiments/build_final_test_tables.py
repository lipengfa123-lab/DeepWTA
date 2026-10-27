#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.final_analysis_common import (
    TARGET_METHODS, canonical_method, scenario_units, summarize_method_frame,
    bootstrap_ci, exact_mcnemar_p, holm_adjust, safe_wilcoxon, numeric,
)

EXCLUDE_TOKENS = (
    "allocation_only", "visual_gallery", "scaling",
    "experiment2", "analysis_exp2", "training",
)


def candidate_csvs(search_root: Path):
    for p in search_root.rglob("*.csv"):
        s = str(p).lower().replace("\\", "/")
        if any(t in s for t in EXCLUDE_TOKENS):
            continue
        name = p.name.lower()
        if "test" not in name and "detail" not in name and "result" not in name:
            continue
        yield p


def inspect_csv(path: Path):
    try:
        f = pd.read_csv(path)
    except Exception:
        return []
    if "method" not in f.columns:
        return []
    f = f.copy()
    f["method"] = f["method"].map(canonical_method)
    rows = []
    for method in TARGET_METHODS:
        g = f[f["method"] == method]
        if g.empty:
            continue
        sid = "scenario_index" if "scenario_index" in g.columns else ("index" if "index" in g.columns else None)
        ncase = int(g[sid].nunique()) if sid else len(g)
        score = ncase * 1000
        s = str(path).lower()
        if "test" in s: score += 500
        if "final" in s or "frozen" in s or "formal" in s: score += 200
        if "smoke" in s or "pilot" in s: score -= 500
        rows.append((method, path, ncase, score))
    return rows


def discover(search_root: Path):
    options = {m: [] for m in TARGET_METHODS}
    for p in candidate_csvs(search_root):
        for item in inspect_csv(p):
            options[item[0]].append(item)
    selected = {}
    for m, xs in options.items():
        if xs:
            xs.sort(key=lambda x: (x[3], x[2], x[1].stat().st_mtime), reverse=True)
            selected[m] = xs[0]
    return selected, options


def load_selected(selected):
    frames = []
    audit = []
    for method, (_, path, ncase, score) in selected.items():
        f = pd.read_csv(path)
        f["method"] = f["method"].map(canonical_method)
        f = f[f["method"] == method].copy()
        frames.append(f)
        audit.append({
            "method": method, "path": str(path.resolve()),
            "raw_rows": len(f), "unique_scenarios": ncase, "selection_score": score,
        })
    return frames, audit


def paired_statistics(units: pd.DataFrame, refs=("DeepWTA-Standard",),
                      bootstrap_samples=10000, seed=2026):
    out = []
    for ref_name in refs:
        ref = units[units.method == ref_name].copy()
        if ref.empty:
            continue
        ref = ref[["scenario_index","hard_feasible","profile_score"]].rename(
            columns={"hard_feasible":"ref_feas","profile_score":"ref_j"})
        for method in TARGET_METHODS:
            if method == ref_name:
                continue
            b = units[units.method == method].copy()
            if b.empty:
                continue
            b = b[["scenario_index","hard_feasible","profile_score"]].rename(
                columns={"hard_feasible":"b_feas","profile_score":"b_j"})
            p = ref.merge(b, on="scenario_index", how="inner")
            if p.empty:
                continue

            both = p[(p.ref_feas > .5) & (p.b_feas > .5)].copy()
            d = (both.b_j - both.ref_j).to_numpy(float)
            mean, lo, hi = bootstrap_ci(d, samples=bootstrap_samples, seed=seed+len(out))
            wp = safe_wilcoxon(d)

            ref_only = int(((p.ref_feas>.5) & ~(p.b_feas>.5)).sum())
            base_only = int((~(p.ref_feas>.5) & (p.b_feas>.5)).sum())
            mp = exact_mcnemar_p(ref_only, base_only)

            out.append({
                "Reference": ref_name,
                "Baseline": method,
                "PairedN": len(p),
                "BothFeasibleN": len(both),
                "DeltaJ_BaselineMinusReference": mean,
                "DeltaJ_CI95_Low": lo,
                "DeltaJ_CI95_High": hi,
                "WilcoxonP_Raw": wp,
                "ReferenceFeasible_BaselineInfeasible": ref_only,
                "ReferenceInfeasible_BaselineFeasible": base_only,
                "McNemarExactP_Raw": mp,
            })

    t = pd.DataFrame(out)
    if t.empty:
        return t
    t["WilcoxonP_Holm"] = np.nan
    t["McNemarExactP_Holm"] = np.nan
    for ref, idx in t.groupby("Reference").groups.items():
        ids = list(idx)
        t.loc[ids, "WilcoxonP_Holm"] = holm_adjust(t.loc[ids,"WilcoxonP_Raw"].to_numpy())
        t.loc[ids, "McNemarExactP_Holm"] = holm_adjust(t.loc[ids,"McNemarExactP_Raw"].to_numpy())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-root", type=Path, default=Path("dataset/test800"))
    ap.add_argument("--output-dir", type=Path, default=Path("reproduced/tables/test800_final"))
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-test-cases", type=int, default=100)
    args = ap.parse_args()

    selected, options = discover(args.search_root)
    missing = [m for m in TARGET_METHODS if m not in selected]
    if missing:
        print("WARNING: no main TEST CSV automatically selected for:", ", ".join(missing))

    frames, audit = load_selected(selected)
    if not frames:
        raise RuntimeError("No main TEST result CSVs were found.")

    raw = pd.concat(frames, ignore_index=True, sort=False)
    units = scenario_units(raw)

    counts = units.groupby("method").scenario_index.nunique().to_dict()
    too_small = {m:n for m,n in counts.items() if n < args.min_test_cases}
    if too_small:
        raise RuntimeError(
            "Selected TEST results look like smoke/pilot data: "
            + ", ".join(f"{m}={n}" for m,n in too_small.items())
        )

    rows = []
    for method in TARGET_METHODS:
        g = units[units.method == method].copy()
        if g.empty:
            continue
        row = {"Method": method}
        row.update(summarize_method_frame(g))
        feas = numeric(g, ["hard_feasible", "feasible"])
        j_series = numeric(g, ["profile_score", "J", "objective", "canonical_objective"])
        j = j_series[feas > .5].dropna().to_numpy()
        mean, lo, hi = bootstrap_ci(j, samples=args.bootstrap_samples, seed=args.seed+len(rows))
        row["J_CI95_Low"] = lo
        row["J_CI95_High"] = hi
        rows.append(row)

    table = pd.DataFrame(rows)
    stats = paired_statistics(
        units, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir/"table_main.csv", index=False)
    stats.to_csv(args.output_dir/"table_paired_statistics.csv", index=False)
    units.to_csv(args.output_dir/"scenario_units.csv", index=False)
    raw.to_csv(args.output_dir/"all_selected_test_rows.csv", index=False)
    (args.output_dir/"selected_test_input_files.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== FINAL TEST MAIN TABLE ===")
    print(table.to_string(index=False))
    print("\n=== PAIRED STATISTICS ===")
    print(stats.to_string(index=False))
    print("\nWrote:")
    print(args.output_dir/"table_main.csv")
    print(args.output_dir/"table_paired_statistics.csv")
    print(args.output_dir/"selected_test_input_files.json")


if __name__ == "__main__":
    main()
