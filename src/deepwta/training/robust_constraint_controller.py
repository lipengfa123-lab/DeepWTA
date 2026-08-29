from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RobustConstraintTargets:
    demand: float
    inventory_pre: float
    satisfaction: float


class RobustConstraintControllerV2:
    """Robust validation-driven controller for DeepWTA hard constraints."""

    def __init__(
        self,
        lambda_demand: float = 12.0,
        lambda_inventory: float = 3.0,
        demand_bounds: tuple[float, float] = (6.0, 96.0),
        inventory_bounds: tuple[float, float] = (1.5, 24.0),
        target_mode: str = "auto",
        fixed_target_demand: float = 2.0e-3,
        fixed_target_inventory: float = 4.0e-4,
        fixed_target_satisfaction: float = 0.985,
        balanced_window: int = 10,
        robust_window: int = 3,
        demand_quantile: float = 0.75,
        inventory_quantile: float = 0.75,
        satisfaction_quantile: float = 0.50,
        demand_target_slack: float = 1.05,
        inventory_target_slack: float = 1.10,
        satisfaction_slack: float = 0.001,
        pressure_beta: float = 0.65,
        hysteresis_low: float = 0.90,
        hysteresis_high: float = 1.10,
        eta_up: float = 0.10,
        eta_down: float = 0.035,
        max_log_step_up: float = 0.12,
        max_log_step_down: float = 0.05,
        update_interval: int = 2,
        full_warmup_epochs: int = 3,
        soft_gate_floor: float = 0.45,
        soft_gate_beta: float = 0.60,
    ):
        if target_mode not in {"auto", "fixed"}:
            raise ValueError("target_mode must be 'auto' or 'fixed'")
        if not 0 < robust_window:
            raise ValueError("robust_window must be positive")
        if not 0 < update_interval:
            raise ValueError("update_interval must be positive")
        if not 0.0 < soft_gate_floor <= 1.0:
            raise ValueError("soft_gate_floor must be in (0,1]")
        if not 0.0 <= pressure_beta < 1.0:
            raise ValueError("pressure_beta must be in [0,1)")
        if not 0.0 <= soft_gate_beta < 1.0:
            raise ValueError("soft_gate_beta must be in [0,1)")
        if not hysteresis_low < 1.0 < hysteresis_high:
            raise ValueError("hysteresis must straddle 1")

        self.lambda_demand = float(lambda_demand)
        self.lambda_inventory = float(lambda_inventory)
        self.demand_bounds = tuple(map(float, demand_bounds))
        self.inventory_bounds = tuple(map(float, inventory_bounds))
        self.target_mode = str(target_mode)

        self.targets = RobustConstraintTargets(
            demand=float(fixed_target_demand),
            inventory_pre=float(fixed_target_inventory),
            satisfaction=float(fixed_target_satisfaction),
        )

        self.balanced_window = int(max(balanced_window, 5))
        self.robust_window = int(robust_window)
        self.demand_quantile = float(demand_quantile)
        self.inventory_quantile = float(inventory_quantile)
        self.satisfaction_quantile = float(satisfaction_quantile)
        self.demand_target_slack = float(demand_target_slack)
        self.inventory_target_slack = float(inventory_target_slack)
        self.satisfaction_slack = float(satisfaction_slack)

        self.pressure_beta = float(pressure_beta)
        self.hysteresis_low = float(hysteresis_low)
        self.hysteresis_high = float(hysteresis_high)
        self.eta_up = float(eta_up)
        self.eta_down = float(eta_down)
        self.max_log_step_up = float(max_log_step_up)
        self.max_log_step_down = float(max_log_step_down)
        self.update_interval = int(update_interval)
        self.full_warmup_epochs = int(full_warmup_epochs)
        self.soft_gate_floor = float(soft_gate_floor)
        self.soft_gate_beta = float(soft_gate_beta)

        self._balanced_history: deque[tuple[float, float, float]] = deque(
            maxlen=self.balanced_window
        )
        self._full_history: deque[tuple[float, float, float]] = deque(
            maxlen=self.robust_window
        )

        self._ema_demand_pressure = 1.0
        self._ema_inventory_pressure = 1.0
        self._soft_gate = 1.0
        self._full_epoch = 0
        self._safe_streak = 0
        self._violation_streak = 0
        self._calibrated = target_mode == "fixed"

    def observe_balanced(self, metrics: Mapping[str, float]) -> None:
        self._balanced_history.append(
            (
                float(metrics["demand"]),
                float(metrics["inventory_pre"]),
                float(metrics["demand_satisfaction_rate"]),
            )
        )

    def calibrate_from_balanced(self) -> RobustConstraintTargets:
        if self.target_mode == "fixed":
            self._calibrated = True
            return self.targets

        if len(self._balanced_history) < 3:
            self._calibrated = True
            return self.targets

        arr = np.asarray(self._balanced_history, dtype=np.float64)
        demand = float(np.quantile(arr[:, 0], self.demand_quantile))
        inventory = float(np.quantile(arr[:, 1], self.inventory_quantile))
        satisfaction = float(
            np.quantile(arr[:, 2], self.satisfaction_quantile)
        )

        self.targets = RobustConstraintTargets(
            demand=max(5.0e-4, demand * self.demand_target_slack),
            inventory_pre=max(
                5.0e-5, inventory * self.inventory_target_slack
            ),
            satisfaction=float(
                np.clip(
                    satisfaction - self.satisfaction_slack,
                    0.95,
                    0.9975,
                )
            ),
        )
        self._calibrated = True
        return self.targets

    def start_full_phase(self) -> RobustConstraintTargets:
        targets = self.calibrate_from_balanced()
        self._full_epoch = 0
        self._full_history.clear()
        self._safe_streak = 0
        self._violation_streak = 0
        self._ema_demand_pressure = 1.0
        self._ema_inventory_pressure = 1.0
        self._soft_gate = 1.0
        return targets

    def observe_full(self, metrics: Mapping[str, float]) -> None:
        self._full_epoch += 1
        self._full_history.append(
            (
                float(metrics["demand"]),
                float(metrics["inventory_pre"]),
                float(metrics["demand_satisfaction_rate"]),
            )
        )

    def robust_metrics(self) -> dict[str, float]:
        if not self._full_history:
            return {
                "demand": self.targets.demand,
                "inventory_pre": self.targets.inventory_pre,
                "demand_satisfaction_rate": self.targets.satisfaction,
            }
        arr = np.asarray(self._full_history, dtype=np.float64)
        return {
            "demand": float(np.median(arr[:, 0])),
            "inventory_pre": float(np.median(arr[:, 1])),
            "demand_satisfaction_rate": float(np.median(arr[:, 2])),
        }

    @staticmethod
    def _ratio(value: float, target: float) -> float:
        return float(value) / max(float(target), 1.0e-12)

    def pressures(self) -> tuple[float, float]:
        if not self._calibrated:
            self.calibrate_from_balanced()

        metrics = self.robust_metrics()
        demand_pressure = self._ratio(
            metrics["demand"], self.targets.demand
        )
        failure = max(
            1.0 - metrics["demand_satisfaction_rate"], 1.0e-6
        )
        target_failure = max(1.0 - self.targets.satisfaction, 1.0e-6)
        satisfaction_pressure = failure / target_failure
        demand_pressure = max(demand_pressure, satisfaction_pressure)
        inventory_pressure = self._ratio(
            metrics["inventory_pre"], self.targets.inventory_pre
        )
        return float(demand_pressure), float(inventory_pressure)

    def _ema_pressure(self, current: float, previous: float) -> float:
        return (
            self.pressure_beta * previous
            + (1.0 - self.pressure_beta) * current
        )

    def _bounded_update(
        self,
        value: float,
        pressure: float,
        bounds: tuple[float, float],
    ) -> float:
        if pressure > self.hysteresis_high:
            exponent = self.eta_up * (
                pressure - self.hysteresis_high
            )
            exponent = min(exponent, self.max_log_step_up)
        elif pressure < self.hysteresis_low:
            exponent = self.eta_down * (
                pressure - self.hysteresis_low
            )
            exponent = max(exponent, -self.max_log_step_down)
        else:
            exponent = 0.0

        updated = float(value) * math.exp(float(exponent))
        return float(np.clip(updated, bounds[0], bounds[1]))

    def _update_soft_gate(self) -> None:
        pressure = max(
            self._ema_demand_pressure,
            self._ema_inventory_pressure,
            1.0,
        )
        desired = float(
            np.clip(
                1.0 / math.sqrt(pressure),
                self.soft_gate_floor,
                1.0,
            )
        )
        self._soft_gate = (
            self.soft_gate_beta * self._soft_gate
            + (1.0 - self.soft_gate_beta) * desired
        )

    def update_if_due(self) -> dict[str, float]:
        if not self._calibrated:
            self.calibrate_from_balanced()

        demand_pressure, inventory_pressure = self.pressures()
        self._ema_demand_pressure = self._ema_pressure(
            demand_pressure, self._ema_demand_pressure
        )
        self._ema_inventory_pressure = self._ema_pressure(
            inventory_pressure, self._ema_inventory_pressure
        )
        self._update_soft_gate()

        max_pressure = max(
            self._ema_demand_pressure, self._ema_inventory_pressure
        )
        if max_pressure < self.hysteresis_low:
            self._safe_streak += 1
            self._violation_streak = 0
        elif max_pressure > self.hysteresis_high:
            self._violation_streak += 1
            self._safe_streak = 0
        else:
            self._safe_streak = 0
            self._violation_streak = 0

        due = (
            self._full_epoch > self.full_warmup_epochs
            and self._full_epoch % self.update_interval == 0
        )
        if due:
            self.lambda_demand = self._bounded_update(
                self.lambda_demand,
                self._ema_demand_pressure,
                self.demand_bounds,
            )
            self.lambda_inventory = self._bounded_update(
                self.lambda_inventory,
                self._ema_inventory_pressure,
                self.inventory_bounds,
            )
            if self._safe_streak >= 2:
                self.lambda_demand = max(
                    self.demand_bounds[0], 0.95 * self.lambda_demand
                )
                self.lambda_inventory = max(
                    self.inventory_bounds[0],
                    0.95 * self.lambda_inventory,
                )

        return self.snapshot()

    def soft_gate(self) -> float:
        return float(self._soft_gate)

    def selection_feasible(
        self, metrics: Mapping[str, float]
    ) -> bool:
        return bool(
            float(metrics["demand"]) <= self.targets.demand
            and float(metrics["inventory_pre"])
            <= self.targets.inventory_pre
            and float(metrics["demand_satisfaction_rate"])
            >= self.targets.satisfaction
        )

    def snapshot(self) -> dict[str, float]:
        robust = self.robust_metrics()
        return {
            "lambda_demand": float(self.lambda_demand),
            "lambda_inventory": float(self.lambda_inventory),
            "target_demand": float(self.targets.demand),
            "target_inventory_pre": float(self.targets.inventory_pre),
            "target_satisfaction": float(self.targets.satisfaction),
            "robust_demand": float(robust["demand"]),
            "robust_inventory_pre": float(robust["inventory_pre"]),
            "robust_satisfaction": float(
                robust["demand_satisfaction_rate"]
            ),
            "ema_demand_pressure": float(self._ema_demand_pressure),
            "ema_inventory_pressure": float(
                self._ema_inventory_pressure
            ),
            "soft_gate": float(self._soft_gate),
            "full_epoch": float(self._full_epoch),
            "safe_streak": float(self._safe_streak),
            "violation_streak": float(self._violation_streak),
            "calibrated": float(self._calibrated),
        }

    def state_dict(self) -> dict:
        return {
            **self.snapshot(),
            "target_mode": self.target_mode,
            "balanced_history": list(self._balanced_history),
            "full_history": list(self._full_history),
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.lambda_demand = float(state["lambda_demand"])
        self.lambda_inventory = float(state["lambda_inventory"])
        self.targets = RobustConstraintTargets(
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
        self._soft_gate = float(state.get("soft_gate", 1.0))
        self._full_epoch = int(state.get("full_epoch", 0))
        self._safe_streak = int(state.get("safe_streak", 0))
        self._violation_streak = int(
            state.get("violation_streak", 0)
        )
        self._calibrated = bool(state.get("calibrated", 1.0))

        self._balanced_history.clear()
        for item in state.get("balanced_history", []):
            self._balanced_history.append(tuple(map(float, item)))

        self._full_history.clear()
        for item in state.get("full_history", []):
            self._full_history.append(tuple(map(float, item)))
