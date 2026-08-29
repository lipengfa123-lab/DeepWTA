from __future__ import annotations

"""Frozen protocol for the staged DeepWTA validation and test programme.

All model, solver, and checkpoint choices are made on validation data.  The
test split is intentionally excluded until ``frozen_for_test`` is set after a
human-readable selection report has been reviewed.
"""

from dataclasses import dataclass


FORMAL_SCALES = ("small", "medium", "large", "xlarge")
FORMAL_MODES = ("standard",)
SELECTION_SPLIT = "val"
FINAL_SPLIT = "test"
TACTICAL_STYLE = "balanced"
SEED = 2026

# Provenance frozen from the retained checkpoint and the supplied
# train_deepwta_t2b.py.  Relative shares are semantic preferences; the 0.77
# budget and the controller soft gate determine their actual training scale.
OFFICIAL_TRAINING_RUN = "t2b_tactical_v2_balanced_seed2026"
OFFICIAL_TRAINING_VARIANT = "T2B_ROBUST_BALANCED"
OFFICIAL_TRAINING_PROFILE = {
    "ammo": 0.32,
    "crossing": 0.28,
    "distance": 0.22,
    "interception": 0.18,
}
OFFICIAL_TACTICAL_BUDGET = 0.77
OFFICIAL_TRAINING_EFFECTIVE_WEIGHTS = {
    name: OFFICIAL_TACTICAL_BUDGET * value
    for name, value in OFFICIAL_TRAINING_PROFILE.items()
}


@dataclass(frozen=True)
class SelectionRule:
    feasible_rate_floor: float = 1.0
    primary_metric: str = "profile_score"
    primary_direction: str = "min"
    runtime_metric: str = "total_runtime_ms"
    # A candidate must improve profile score by at least this relative amount
    # before a runtime regression is accepted.
    min_relative_quality_gain: float = 0.002


SELECTION_RULE = SelectionRule()

# Contribution ablation, solver-parameter tuning, and downstream checkpoint
# selection have now been completed on validation data.  The retained choices
# are recorded in ``selected_solver_parameters.json`` and
# ``selected_checkpoint.json``; the test split may now be evaluated once.
FROZEN_FOR_TEST = True
