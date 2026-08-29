from __future__ import annotations

import numpy as np

from .schema import WTAInstance, WTASolution



def canonicalize_nonredundant_allocation(
    allocation: np.ndarray,
    selected_scheme: np.ndarray,
    scheme_demand: np.ndarray,
    intercept_prob: np.ndarray,
    *,
    tol: float = 1e-7,
) -> np.ndarray:
    """Array-only form used by native baseline evaluators."""
    a = np.maximum(np.rint(np.asarray(allocation)).astype(np.int64), 0)
    selected = np.asarray(selected_scheme, dtype=np.int64)
    k_count, _n_count, m_count = a.shape
    if selected.shape != (m_count,):
        return a
    for m in range(m_count):
        l = int(selected[m])
        if l < 0 or l >= np.asarray(scheme_demand).shape[0]:
            continue
        for k in range(k_count):
            demand = float(scheme_demand[l, k, m])
            if demand <= tol:
                a[k, :, m] = 0
                continue
            survival = 1.0 - np.asarray(intercept_prob[k, :, m], dtype=np.float64)
            delivered = float(np.dot(a[k, :, m], survival))
            if delivered + tol < demand:
                continue
            while True:
                active = np.flatnonzero(a[k, :, m] > 0)
                removable = [
                    int(n) for n in active.tolist()
                    if delivered - float(survival[n]) + tol >= demand
                ]
                if not removable:
                    break
                n = min(removable, key=lambda x: (float(survival[x]), -int(x)))
                a[k, n, m] -= 1
                delivered -= float(survival[n])
    return a

def canonicalize_nonredundant_solution(
    instance: WTAInstance,
    solution: WTASolution,
    *,
    tol: float = 1e-7,
) -> WTASolution:
    """Remove individually redundant integer shots using feasibility only.

    Formal v5 evaluation normalizes every solution to the same non-redundant output
    convention.  For each selected (weapon,target) demand, any shot whose removal
    still meets effective-hit demand is deleted.  Among simultaneously removable
    shots we remove the *lowest-survival* shot first; this rule depends only on the
    physical feasibility equation, not distance/crossing/profile weights.

    Infeasible pairs are never repaired or hidden: if delivered effective hits are
    below demand, their allocation is left unchanged and the exact evaluator will
    still mark the solution infeasible.
    """
    selected = np.asarray(solution.selected_scheme, dtype=np.int64).copy()
    allocation_raw = np.asarray(solution.allocation)
    if not np.all(np.isfinite(allocation_raw)):
        return solution
    allocation = np.rint(allocation_raw).astype(np.int64, copy=True)
    allocation = np.maximum(allocation, 0)
    k_count, n_count, m_count = allocation.shape
    if selected.shape != (m_count,):
        return solution

    for m in range(m_count):
        l = int(selected[m])
        if l < 0 or l >= instance.num_schemes:
            continue
        for k in range(k_count):
            demand = float(instance.scheme_demand[l, k, m])
            if demand <= tol:
                # No selected demand: all shots for this weapon-target pair are redundant.
                allocation[k, :, m] = 0
                continue
            survival = 1.0 - instance.intercept_prob[k, :, m].astype(np.float64)
            delivered = float(np.dot(allocation[k, :, m], survival))
            if delivered + tol < demand:
                continue
            while True:
                active = np.flatnonzero(allocation[k, :, m] > 0)
                if active.size == 0:
                    break
                removable = [
                    int(n)
                    for n in active.tolist()
                    if delivered - float(survival[n]) + tol >= demand
                ]
                if not removable:
                    break
                # Lowest survival is least useful to the hard-demand constraint.
                # Higher base index is only a deterministic tie-breaker.
                n = min(removable, key=lambda x: (float(survival[x]), -int(x)))
                allocation[k, n, m] -= 1
                delivered -= float(survival[n])

    return WTASolution(
        selected_scheme=selected,
        allocation=allocation,
        method=solution.method,
        stage=solution.stage,
    )
