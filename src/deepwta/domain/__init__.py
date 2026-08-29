"""Canonical problem, solution and validation contracts."""

from .problem import SolverResult, SolverStats, WTAInstance, WTASolution
from .validation import CROSSING_POLICY, validate_fixed_plan, validate_solution

__all__ = [
    "CROSSING_POLICY",
    "SolverResult",
    "SolverStats",
    "WTAInstance",
    "WTASolution",
    "validate_fixed_plan",
    "validate_solution",
]

