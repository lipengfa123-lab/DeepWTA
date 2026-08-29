"""Stable import location for WTA domain entities.

The concrete definitions currently live in ``benchmark.schema`` so historical
checkpoints and imports retain their module identity. New code should import from
this module; the compatibility implementation can then be migrated independently.
"""

from deepwta.benchmark.schema import SolverResult, SolverStats, WTAInstance, WTASolution

__all__ = ["SolverResult", "SolverStats", "WTAInstance", "WTASolution"]

