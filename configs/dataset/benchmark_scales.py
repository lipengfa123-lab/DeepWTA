from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkScale:
    name: str
    total_weapons: int
    targets: int
    bases: int

    @property
    def dataset_name(self) -> str:
        return f"wta_{self.name}"


BENCHMARK_SCALES = {
    "small": BenchmarkScale("small", total_weapons=50, targets=20, bases=20),
    "medium": BenchmarkScale("medium", total_weapons=100, targets=50, bases=50),
    "large": BenchmarkScale("large", total_weapons=200, targets=100, bases=100),
    "xlarge": BenchmarkScale("xlarge", total_weapons=500, targets=200, bases=200),
}


def get_benchmark_scale(name: str) -> BenchmarkScale:
    try:
        return BENCHMARK_SCALES[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown benchmark scale {name!r}; available={sorted(BENCHMARK_SCALES)}"
        ) from exc
