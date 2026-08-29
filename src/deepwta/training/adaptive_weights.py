from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


SOFT_KEYS = (
    "ammo",
    "crossing",
    "distance",
    "interception",
)


@dataclass
class ConstraintTargets:
    demand: float
    inventory_pre: float
    satisfaction: float


class AdaptiveConstraintController:
    """Bounded multiplicative dual controller for hard constraints.

    Demand and inventory are treated as constraints, not tactical preferences.
    The controller increases their multipliers when validation feasibility
    deteriorates, and relaxes them slowly when feasibility is comfortably met.

    In ``target_mode='auto'`` the targets are calibrated from the last few
    BALANCED validation epochs, so the FULL stage is asked to preserve roughly
    the feasibility quality already learned by the curriculum.
    """

    def __init__(
        self,
        lambda_demand: float = 12.0,
        lambda_inventory: float = 3.0,
        demand_bounds: tuple[float, float] = (6.0, 128.0),
        inventory_bounds: tuple[float, float] = (1.5, 32.0),
        eta: float = 0.12,
        pressure_beta: float = 0.60,
        max_log_step: float = 0.20,
        min_soft_gate: float = 0.35,
        target_mode: str = "auto",
        target_demand: float = 1.0e-3,
        target_inventory: float = 1.0e-4,
        target_satisfaction: float = 0.995,
        auto_window: int = 5,
    ):
        if target_mode not in {"auto", "fixed"}:
            raise ValueError("target_mode must be 'auto' or 'fixed'")
        if not 0.0 <= pressure_beta < 1.0:
            raise ValueError("pressure_beta must satisfy 0 <= beta < 1")
        if eta <= 0.0:
            raise ValueError("eta must be positive")
        if not 0.0 < min_soft_gate <= 1.0:
            raise ValueError("min_soft_gate must be in (0,1]")

        self.lambda_demand = float(lambda_demand)
        self.lambda_inventory = float(lambda_inventory)
        self.demand_bounds = tuple(map(float, demand_bounds))
        self.inventory_bounds = tuple(map(float, inventory_bounds))
        self.eta = float(eta)
        self.pressure_beta = float(pressure_beta)
        self.max_log_step = float(max_log_step)
        self.min_soft_gate = float(min_soft_gate)
        self.target_mode = str(target_mode)
        self.targets = ConstraintTargets(
            demand=float(target_demand),
            inventory_pre=float(target_inventory),
            satisfaction=float(target_satisfaction),
        )
        self.auto_window = int(max(auto_window, 3))

        self._balanced_history: deque[tuple[float, float, float]] = deque(
            maxlen=self.auto_window
        )
        self._ema_demand_pressure = 1.0
        self._ema_inventory_pressure = 1.0
        self._calibrated = target_mode == "fixed"

    def observe_balanced(self, metrics: Mapping[str, float]) -> None:
        self._balanced_history.append(
            (
                float(metrics["demand"]),
                float(metrics["inventory_pre"]),
                float(metrics["demand_satisfaction_rate"]),
            )
        )

    def calibrate_from_balanced(self) -> ConstraintTargets:
        """Set FULL targets from recent BALANCED validation quality."""
        if self.target_mode == "fixed":
            self._calibrated = True
            return self.targets

        if not self._balanced_history:
            self._calibrated = True
            return self.targets

        arr = np.asarray(self._balanced_history, dtype=float)
        demand_med = float(np.median(arr[:, 0]))
        inventory_med = float(np.median(arr[:, 1]))
        sat_med = float(np.median(arr[:, 2]))

        # Preserve approximately the feasibility quality achieved in BALANCED,
        # while avoiding absurdly tiny targets caused by a lucky validation epoch.
        self.targets = ConstraintTargets(
            demand=max(5.0e-4, 1.25 * demand_med),
            inventory_pre=max(5.0e-5, 1.50 * inventory_med),
            satisfaction=min(
                0.9975,
                max(0.9900, sat_med - 0.003),
            ),
        )
        self._calibrated = True
        return self.targets

    @staticmethod
    def _safe_ratio(value: float, target: float) -> float:
        return float(value) / max(float(target), 1.0e-12)

    def pressure(self, metrics: Mapping[str, float]) -> tuple[float, float]:
        if not self._calibrated:
            self.calibrate_from_balanced()

        demand_ratio = self._safe_ratio(
            float(metrics["demand"]),
            self.targets.demand,
        )

        sat = float(metrics["demand_satisfaction_rate"])
        sat_gap = max(self.targets.satisfaction - sat, 0.0)
        sat_scale = max(1.0 - self.targets.satisfaction, 1.0e-3)
        sat_pressure = 1.0 + sat_gap / sat_scale

        demand_pressure = max(demand_ratio, sat_pressure)
        inventory_pressure = self._safe_ratio(
            float(metrics["inventory_pre"]),
            self.targets.inventory_pre,
        )
        return demand_pressure, inventory_pressure

    def _update_one(
        self,
        value: float,
        pressure: float,
        bounds: tuple[float, float],
    ) -> float:
        exponent = self.eta * (float(pressure) - 1.0)
        exponent = float(
            np.clip(
                exponent,
                -self.max_log_step,
                self.max_log_step,
            )
        )
        updated = float(value) * math.exp(exponent)
        return float(np.clip(updated, bounds[0], bounds[1]))

    def update(self, metrics: Mapping[str, float]) -> dict[str, float]:
        demand_pressure, inventory_pressure = self.pressure(metrics)

        self._ema_demand_pressure = (
            self.pressure_beta * self._ema_demand_pressure
            + (1.0 - self.pressure_beta) * demand_pressure
        )
        self._ema_inventory_pressure = (
            self.pressure_beta * self._ema_inventory_pressure
            + (1.0 - self.pressure_beta) * inventory_pressure
        )

        self.lambda_demand = self._update_one(
            self.lambda_demand,
            self._ema_demand_pressure,
            self.demand_bounds,
        )
        self.lambda_inventory = self._update_one(
            self.lambda_inventory,
            self._ema_inventory_pressure,
            self.inventory_bounds,
        )
        return self.snapshot()

    def soft_gate(self) -> float:
        """Automatic global soft-objective attenuation under constraint stress."""
        pressure = max(
            self._ema_demand_pressure,
            self._ema_inventory_pressure,
            1.0,
        )
        gate = 1.0 / math.sqrt(pressure)
        return float(np.clip(gate, self.min_soft_gate, 1.0))

    def selection_feasible(self, metrics: Mapping[str, float]) -> bool:
        if not self._calibrated:
            self.calibrate_from_balanced()
        return bool(
            float(metrics["demand"]) <= self.targets.demand
            and float(metrics["inventory_pre"]) <= self.targets.inventory_pre
            and float(metrics["demand_satisfaction_rate"])
            >= self.targets.satisfaction
        )

    def snapshot(self) -> dict[str, float]:
        return {
            "lambda_demand": float(self.lambda_demand),
            "lambda_inventory": float(self.lambda_inventory),
            "target_demand": float(self.targets.demand),
            "target_inventory_pre": float(self.targets.inventory_pre),
            "target_satisfaction": float(self.targets.satisfaction),
            "ema_demand_pressure": float(self._ema_demand_pressure),
            "ema_inventory_pressure": float(self._ema_inventory_pressure),
            "soft_gate": float(self.soft_gate()),
            "calibrated": float(self._calibrated),
        }

    def state_dict(self) -> dict:
        return {
            **self.snapshot(),
            "target_mode": self.target_mode,
            "balanced_history": list(self._balanced_history),
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.lambda_demand = float(state["lambda_demand"])
        self.lambda_inventory = float(state["lambda_inventory"])
        self.targets = ConstraintTargets(
            demand=float(state["target_demand"]),
            inventory_pre=float(state["target_inventory_pre"]),
            satisfaction=float(state["target_satisfaction"]),
        )
        self._ema_demand_pressure = float(
            state.get("ema_demand_pressure", 1.0)
        )
        self._ema_inventory_pressure = float(
            state.get("ema_inventory_pressure", 1.0)
        )
        self._calibrated = bool(state.get("calibrated", 1.0))
        self._balanced_history.clear()
        for item in state.get("balanced_history", []):
            self._balanced_history.append(tuple(map(float, item)))


class DynamicSoftWeightController:
    """Dynamic Weight Averaging (DWA) with optional semantic prior.

    DWA uses relative loss decrease rates, so absolute units largely cancel.
    The total tactical soft budget remains fixed; only its distribution across
    ammo/crossing/distance/interception changes automatically.

    ``prior_mode='uniform'`` is the least hand-crafted setting.
    ``prior_mode='semantic'`` uses the current DeepWTA proportions as a weak
    prior while still allowing DWA to adapt.
    """

    def __init__(
        self,
        budget: float = 0.77,
        temperature: float = 2.0,
        smoothing: float = 0.50,
        ratio_clip: tuple[float, float] = (0.50, 1.50),
        prior_mode: str = "uniform",
    ):
        if budget <= 0.0:
            raise ValueError("budget must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must satisfy 0 <= smoothing < 1")
        if prior_mode not in {"uniform", "semantic"}:
            raise ValueError("prior_mode must be uniform or semantic")

        self.budget = float(budget)
        self.temperature = float(temperature)
        self.smoothing = float(smoothing)
        self.ratio_clip = tuple(map(float, ratio_clip))
        self.prior_mode = str(prior_mode)

        if prior_mode == "uniform":
            prior = np.ones(len(SOFT_KEYS), dtype=np.float64)
        else:
            # Existing T1 tactical proportions, normalized within the four
            # tactical losses.  This is a weak prior, not a fixed weight vector.
            prior = np.asarray(
                [0.22, 0.35, 0.10, 0.10],
                dtype=np.float64,
            )

        prior = prior / prior.sum()
        self._prior = prior
        self._current = self.budget * prior
        self._history = {
            key: deque(maxlen=2)
            for key in SOFT_KEYS
        }

    def observe(self, metrics: Mapping[str, float]) -> dict[str, float]:
        ratios = []
        ready = True

        for key in SOFT_KEYS:
            value = max(float(metrics[key]), 1.0e-12)
            history = self._history[key]
            if len(history) >= 1:
                ratio = value / max(history[-1], 1.0e-12)
            else:
                ratio = 1.0
                ready = False
            ratios.append(ratio)
            history.append(value)

        if ready:
            ratio = np.clip(
                np.asarray(ratios, dtype=np.float64),
                self.ratio_clip[0],
                self.ratio_clip[1],
            )

            logits = ratio / self.temperature
            if self.prior_mode == "semantic":
                logits = logits + np.log(self._prior + 1.0e-12)

            logits = logits - np.max(logits)
            target = np.exp(logits)
            target = target / target.sum()
            target = self.budget * target

            self._current = (
                self.smoothing * self._current
                + (1.0 - self.smoothing) * target
            )
            self._current *= self.budget / self._current.sum()

        return self.weights()

    def weights(self, gate: float = 1.0) -> dict[str, float]:
        gate = float(np.clip(gate, 0.0, 1.0))
        return {
            key: float(value * gate)
            for key, value in zip(SOFT_KEYS, self._current)
        }

    def snapshot(self) -> dict[str, float]:
        result = {
            f"weight_{key}": float(value)
            for key, value in zip(SOFT_KEYS, self._current)
        }
        result["soft_budget"] = self.budget
        return result

    def state_dict(self) -> dict:
        return {
            "budget": self.budget,
            "temperature": self.temperature,
            "smoothing": self.smoothing,
            "ratio_clip": self.ratio_clip,
            "prior_mode": self.prior_mode,
            "current": self._current.tolist(),
            "history": {
                key: list(values)
                for key, values in self._history.items()
            },
        }

    def load_state_dict(self, state: Mapping) -> None:
        self._current = np.asarray(
            state["current"],
            dtype=np.float64,
        )
        for key in SOFT_KEYS:
            self._history[key].clear()
            for value in state.get("history", {}).get(key, []):
                self._history[key].append(float(value))
