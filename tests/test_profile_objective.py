from __future__ import annotations

from deepwta.evaluation.profile_objective import (
    DOWNSTREAM_REFERENCE_SCALES,
    TACTICAL_PROFILES,
    profile_score_breakdown,
)


def test_profiles_sum_to_one():
    for name, weights in TACTICAL_PROFILES.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-12, name


def test_score_is_normalized_weighted_sum():
    metrics = {
        "ammo": 100.0,
        "ammo_norm": 1.30,
        "raw_ammo_capacity_lb": 100.0,
        "ammo_ratio_to_raw_lb": 1.0,
        "crossing_pair_rate": DOWNSTREAM_REFERENCE_SCALES["crossing_pair_rate"],
        "avg_distance": DOWNSTREAM_REFERENCE_SCALES["avg_distance"],
        "avg_interception": DOWNSTREAM_REFERENCE_SCALES["avg_interception"],
    }
    for style in TACTICAL_PROFILES:
        result = profile_score_breakdown(metrics, style)
        assert abs(result["profile_score"] - 1.0) < 1e-12
        assert "legacy_profile_score" in result


def test_style_changes_preference_without_scale_domination():
    conflict_good = {
        "ammo": 160.0,
        "ammo_norm": 1.30,
        "raw_ammo_capacity_lb": 100.0,
        "ammo_ratio_to_raw_lb": 1.60,
        "crossing_pair_rate": 0.03,
        "avg_distance": 0.52,
        "avg_interception": 0.132,
    }
    resource_good = {
        "ammo": 100.0,
        "ammo_norm": 1.30,
        "raw_ammo_capacity_lb": 100.0,
        "ammo_ratio_to_raw_lb": 1.00,
        "crossing_pair_rate": 0.12,
        "avg_distance": 0.52,
        "avg_interception": 0.132,
    }
    assert (
        profile_score_breakdown(conflict_good, "conflict_averse")["profile_score"]
        < profile_score_breakdown(resource_good, "conflict_averse")["profile_score"]
    )
    assert (
        profile_score_breakdown(resource_good, "resource_efficient")["profile_score"]
        < profile_score_breakdown(conflict_good, "resource_efficient")["profile_score"]
    )


def main():
    test_profiles_sum_to_one()
    test_score_is_normalized_weighted_sum()
    test_style_changes_preference_without_scale_domination()
    print("PASS: operational-normalized tactical profile objective tests")


if __name__ == "__main__":
    main()
