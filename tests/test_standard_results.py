from __future__ import annotations

import pandas as pd
import pytest

from scripts.reviewer2.summarize_standard_results import summary_row, validate_rows


def rows(method: str = "DeepWTA-Standard") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_index": [0, 1],
            "scenario_id": ["case-0", "case-1"],
            "method": [method, method],
            "hard_feasible": [1.0, 1.0],
            "profile_score": [1.0, 1.2],
            "ammo_ratio_to_raw_lb": [1.01, 1.03],
            "crossing_pair_rate": [0.1, 0.2],
            "avg_distance": [0.4, 0.6],
            "avg_interception": [0.12, 0.14],
            "total_runtime_ms": [10.0, 30.0],
        }
    )


def test_standard_summary_contains_only_standard_rows() -> None:
    frame = rows()
    validate_rows(frame, min_cases=2)
    result = summary_row(frame, bootstrap_samples=100, seed=7)
    assert result["Method"] == "DeepWTA-Standard"
    assert result["N"] == 2
    assert result["Feasible"] == 1.0
    assert result["J"] == pytest.approx(1.1)
    assert result["RuntimeMedianMs"] == pytest.approx(20.0)


def test_standard_summary_rejects_other_methods() -> None:
    with pytest.raises(ValueError, match="expected only"):
        validate_rows(rows("Other-Method"), min_cases=1)
