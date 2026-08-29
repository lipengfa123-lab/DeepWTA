from __future__ import annotations

from typing import Any

from deepwta.evaluation.exact_metrics import scenario_metrics as _scenario_metrics
from deepwta.evaluation.profile_objective import (
    attach_profile_reference_metrics,
    profile_score_breakdown,
)

from .schema import WTAInstance, WTASolution


def evaluate_wta_solution(
    instance: WTAInstance,
    solution: WTASolution,
    tol: float = 1e-7,
    tactical_style: str = "balanced",
) -> dict[str, Any]:
    """All methods terminate at this exact evaluator."""
    metrics = _scenario_metrics(
        solution.allocation,
        solution.selected_scheme,
        instance.inventory,
        instance.scheme_demand,
        instance.damage_level,
        instance.damage_threshold,
        instance.intercept_prob,
        instance.reachable,
        instance.distance,
        instance.base_coords,
        instance.target_coords,
        tol=tol,
    )
    metrics = dict(metrics)
    metrics = attach_profile_reference_metrics(
        metrics,
        B=instance.inventory,
        U=instance.scheme_demand,
        G=instance.damage_level,
        G_th=instance.damage_threshold,
        Q=instance.intercept_prob,
        X=instance.reachable,
        tol=tol,
    )
    metrics.update(profile_score_breakdown(metrics, tactical_style))
    metrics["method"] = solution.method
    metrics["stage"] = solution.stage
    metrics["instance_id"] = instance.instance_id
    metrics["seed"] = instance.seed
    metrics["crossing_policy"] = "soft_tactical_objective"
    return metrics
