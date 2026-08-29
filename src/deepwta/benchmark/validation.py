from __future__ import annotations

from typing import Any

from deepwta.utils.validator import validate_solution as _canonical_validate_solution

from .schema import WTAInstance, WTASolution


CROSSING_POLICY = "soft_tactical_objective"


def validate_wta_solution(
    instance: WTAInstance,
    solution: WTASolution,
    tol: float = 1e-7,
) -> dict[str, Any]:
    """Use the project's single exact validator; do not duplicate constraint logic."""
    instance.check_shapes()
    report = _canonical_validate_solution(
        solution.allocation,
        solution.selected_scheme,
        instance.inventory,
        instance.scheme_demand,
        instance.damage_level,
        instance.damage_threshold,
        instance.intercept_prob,
        instance.reachable,
        instance.base_coords,
        instance.target_coords,
        tol=tol,
    )
    report = dict(report)
    report["crossing_policy"] = CROSSING_POLICY
    report["crossing_is_soft_objective"] = True
    return report
