from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# Frozen T2b tactical soft-objective budget.
#
# The four profile values below are *relative shares* and therefore sum to 1.0.
# During FULL_OPT they are multiplied by this constant budget and by the
# existing RobustConstraintControllerV2 soft gate. Keeping 0.77 frozen preserves
# the original T2b hard-vs-soft scale, so the ablation changes only the relative
# tactical preference.
TACTICAL_BUDGET = 0.77
_PROFILE_SUM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class TacticalProfile:
    name: str
    ammo: float
    crossing: float
    distance: float
    interception: float
    description: str

    @property
    def total(self) -> float:
        return self.ammo + self.crossing + self.distance + self.interception

    def as_dict(self) -> Dict[str, float]:
        """Normalized relative tactical shares; sum must be exactly 1.0."""
        return {
            "ammo": self.ammo,
            "crossing": self.crossing,
            "distance": self.distance,
            "interception": self.interception,
        }

    def effective_dict(self) -> Dict[str, float]:
        """Frozen-budget weights before the existing controller soft gate."""
        return {
            key: TACTICAL_BUDGET * value
            for key, value in self.as_dict().items()
        }


TACTICAL_PROFILES = {
    "balanced": TacticalProfile(
        name="balanced",
        ammo=0.32,
        crossing=0.28,
        distance=0.22,
        interception=0.18,
        description=(
            "Balanced tactical ablation profile: moderate ammunition and "
            "crossing pressure with stronger distance/risk coverage than the "
            "legacy T2b balanced profile."
        ),
    ),
    "conflict_averse": TacticalProfile(
        name="conflict_averse",
        ammo=0.20,
        crossing=0.45,
        distance=0.18,
        interception=0.17,
        description=(
            "Conflict-averse profile: emphasizes route-conflict reduction while "
            "retaining non-zero resource, distance and interception pressure."
        ),
    ),
    "resource_efficient": TacticalProfile(
        name="resource_efficient",
        ammo=0.45,
        crossing=0.18,
        distance=0.22,
        interception=0.15,
        description=(
            "Resource-efficient profile: emphasizes ammunition efficiency while "
            "keeping explicit conflict, distance and interception objectives."
        ),
    ),
    "time_risk_balanced": TacticalProfile(
        name="time_risk_balanced",
        ammo=0.20,
        crossing=0.18,
        distance=0.32,
        interception=0.30,
        description=(
            "Time-risk balanced profile: emphasizes distance/timeliness and "
            "interception risk, with lower ammunition/conflict pressure."
        ),
    ),
}


def validate_tactical_profiles() -> None:
    required = {
        "balanced",
        "conflict_averse",
        "resource_efficient",
        "time_risk_balanced",
    }
    if set(TACTICAL_PROFILES) != required:
        raise RuntimeError(
            "TACTICAL_PROFILES must contain exactly: "
            + ", ".join(sorted(required))
        )

    for name, profile in TACTICAL_PROFILES.items():
        values = profile.as_dict()
        if any(value < 0.0 for value in values.values()):
            raise RuntimeError(f"profile {name!r} contains a negative weight")
        if abs(profile.total - 1.0) > _PROFILE_SUM_TOLERANCE:
            raise RuntimeError(
                f"profile {name!r} has relative-weight sum {profile.total:.16g}; "
                "expected 1.0"
            )

        effective_total = sum(profile.effective_dict().values())
        if abs(effective_total - TACTICAL_BUDGET) > _PROFILE_SUM_TOLERANCE:
            raise RuntimeError(
                f"profile {name!r} has effective tactical budget "
                f"{effective_total:.16g}; expected {TACTICAL_BUDGET:.16g}"
            )


validate_tactical_profiles()


def get_tactical_profile(name: str) -> TacticalProfile:
    try:
        return TACTICAL_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown tactical profile {name!r}; "
            f"available={sorted(TACTICAL_PROFILES)}"
        ) from exc
