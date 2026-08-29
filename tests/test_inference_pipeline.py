from __future__ import annotations

import numpy as np

from deepwta.evaluation.inference_pipeline import select_level_feasible_schemes


def test_select_level_feasible_schemes_masks_invalid_high_probability_choice() -> None:
    # probs [M,L], damage [L,M]
    probs = np.array([[0.8, 0.15, 0.05], [0.1, 0.7, 0.2]])
    damage = np.array([[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]])
    threshold = np.array([2.0, 2.0])
    selected = select_level_feasible_schemes(probs, damage, threshold)
    np.testing.assert_array_equal(selected, np.array([1, 2]))


def test_select_level_feasible_schemes_rejects_missing_valid_choice() -> None:
    probs = np.array([[1.0, 0.0]])
    damage = np.array([[1.0], [1.0]])
    threshold = np.array([2.0])
    try:
        select_level_feasible_schemes(probs, damage, threshold)
    except ValueError as exc:
        assert "no damage-level-feasible scheme" in str(exc)
    else:
        raise AssertionError("expected ValueError")

