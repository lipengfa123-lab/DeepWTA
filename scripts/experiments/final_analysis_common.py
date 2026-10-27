from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_METHODS = [
    "DeepWTA-Standard",
    "FRA-DC",
    "SA-MC",
    "GA-MC",
    "MultiHead-DQN-E2E",
    "PointerNet-AC-E2E",
]

CLASSICAL = {"FRA-DC", "SA-MC", "GA-MC"}

ALIASES = {
    "DeepWTA": "DeepWTA-Standard",
    "DeepWTA-Standard": "DeepWTA-Standard",
    "FRA-DC": "FRA-DC",
    "SA": "SA-MC",
    "SA-MC": "SA-MC",
    "GA": "GA-MC",
    "GA-MC": "GA-MC",
    "MultiHead-DQN-E2E": "MultiHead-DQN-E2E",
    "DQN-E2E": "MultiHead-DQN-E2E",
    "PointerNet-AC-E2E": "PointerNet-AC-E2E",
    "Pointer-E2E": "PointerNet-AC-E2E",
}


def canonical_method(x) -> str:
    s = str(x)
    return ALIASES.get(s, s)


def first_existing_col(df: pd.DataFrame, names: Iterable[str]):
    for n in names:
        if n in df.columns:
            return n
    return None


def numeric(df, names, default=np.nan):
    c = first_existing_col(df, names)
    if c is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[c], errors="coerce")


def bootstrap_ci(values, samples=10000, seed=2026):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    out = np.empty(samples)
    block = 500
    p = 0
    while p < samples:
        n = min(block, samples-p)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        out[p:p+n] = np.mean(x[idx], axis=1)
        p += n
    lo, hi = np.quantile(out, [0.025, 0.975])
    return mean, float(lo), float(hi)


def exact_mcnemar_p(a: int, b: int) -> float:
    n = int(a) + int(b)
    if n == 0:
        return 1.0
    k = min(int(a), int(b))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m-rank)*p[idx])
        running = max(running, adj)
        out[idx] = running
    return out


def safe_wilcoxon(d):
    x = np.asarray(d, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.allclose(x, 0):
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(x).pvalue)
    except Exception:
        return np.nan


def scenario_units(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    f = df.copy()
    if "method" not in f.columns:
        raise ValueError("No method column found.")
    f["method"] = f["method"].map(canonical_method)

    canonical_aliases = {
        "hard_feasible": [
            "hard_feasible", "feasible", "is_feasible", "hard_feasible_rate",
            "feasibility", "feasibility_rate",
        ],
        "profile_score": [
            "profile_score", "J", "objective", "canonical_objective",
            "terminal_cost",
        ],
        "WR": [
            "WR", "ammo_ratio", "wr",
        ],
        "TCR": [
            "TCR", "tcr",
        ],
        "avg_distance": [
            "avg_distance", "average_distance", "Timeliness", "mean_distance",
        ],
        "avg_interception": [
            "avg_interception", "average_interception", "Inter. ratio",
            "mean_interception",
        ],
        "total_runtime_ms": [
            "total_runtime_ms", "runtime_ms", "median_runtime_ms",
            "elapsed_ms", "wallclock_ms",
        ],
    }
    for dst, names in canonical_aliases.items():
        if dst in f.columns:
            continue
        c = first_existing_col(f, names)
        if c is not None:
            f[dst] = pd.to_numeric(f[c], errors="coerce")

    sid = first_existing_col(f, ["scenario_index", "index", "case_index"])
    if sid is None:
        if "scenario_id" in f.columns:
            codes, _ = pd.factorize(f["scenario_id"].astype(str), sort=False)
            f["scenario_index"] = codes.astype(int)
        else:
            raise ValueError("No scenario_index/index/case_index/scenario_id column found.")
    elif sid != "scenario_index":
        f["scenario_index"] = f[sid]

    if "scenario_id" not in f:
        f["scenario_id"] = f["scenario_index"].map(lambda x: f"case_{int(x):06d}")

    keys = ["method", "scenario_index", "scenario_id"]
    if "scale" in f:
        keys.insert(0, "scale")

    numeric_cols = f.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    ignore = {"scenario_index", "method_seed", "solver_seed", "scenario_seed", "seed"}
    agg_cols = [c for c in numeric_cols if c not in ignore and c not in keys]

    if "method_seed" in f.columns or f.duplicated(keys).any():
        g = f.groupby(keys, as_index=False, dropna=False)[agg_cols].mean()
    else:
        g = f[keys + [c for c in agg_cols if c not in keys]].copy()

    return g

def summarize_method_frame(df: pd.DataFrame) -> dict:
    feasible = numeric(df, ["hard_feasible", "feasible"])
    fg = df[feasible > 0.5].copy()

    j_all = numeric(df, ["profile_score", "J", "objective", "canonical_objective"])
    j_feas = numeric(fg, ["profile_score", "J", "objective", "canonical_objective"])
    ammo = numeric(df, ["WR", "ammo_ratio"])
    tcr = numeric(df, ["TCR"])
    dist = numeric(df, ["avg_distance", "average_distance", "Timeliness"])
    inter = numeric(df, ["avg_interception", "average_interception", "Inter. ratio"])
    runtime = numeric(df, ["total_runtime_ms", "runtime_ms", "median_runtime_ms"])

    return {
        "N": int(df["scenario_index"].nunique()) if "scenario_index" in df else len(df),
        "Feasible": float(feasible.mean()) if feasible.notna().any() else np.nan,
        "J_FeasibleMean": float(j_feas.mean()) if j_feas.notna().any() else np.nan,
        "J_AllMean": float(j_all.mean()) if j_all.notna().any() else np.nan,
        "WR": float(ammo.mean()) if ammo.notna().any() else np.nan,
        "TCR": float(tcr.mean()) if tcr.notna().any() else np.nan,
        "Timeliness": float(dist.mean()) if dist.notna().any() else np.nan,
        "Inter. ratio": float(inter.mean()) if inter.notna().any() else np.nan,
        "RuntimeMeanMs": float(runtime.mean()) if runtime.notna().any() else np.nan,
        "RuntimeMedianMs": float(runtime.median()) if runtime.notna().any() else np.nan,
        "RuntimeP95Ms": float(runtime.quantile(.95)) if runtime.notna().any() else np.nan,
    }
