from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepwta.mechanisms.conflict_index import LazyConflictIndex


Route = tuple[int, int]


@dataclass(frozen=True)
class TacticalWeights:
    ammo: float = 10.0
    crossing: float = 80.0
    distance: float = 30.0
    interception: float = 10.0


class IncrementalRepairState:
    """Exact incremental integer-repair state.

    Maintains hard feasibility and tactical aggregates without rescanning the
    full K x N x M tensor after each move.
    """

    def __init__(
        self,
        A: np.ndarray,
        scheme_idx: np.ndarray,
        B: np.ndarray,
        U: np.ndarray,
        Q: np.ndarray,
        X: np.ndarray,
        distance_metric: np.ndarray,
        conflict_index: LazyConflictIndex,
        *,
        weights: TacticalWeights = TacticalWeights(),
        tol: float = 1e-7,
    ):
        self.A = np.asarray(A, dtype=np.int64).copy()
        self.scheme_idx = np.asarray(scheme_idx, dtype=np.int64).copy()
        self.B = np.maximum(
            np.floor(np.asarray(B, dtype=np.float64) + tol),
            0.0,
        ).astype(np.int64)
        self.U = np.asarray(U, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.X = np.asarray(X, dtype=np.float64)
        self.distance_metric = np.asarray(distance_metric, dtype=np.float64)
        self.conflict_index = conflict_index
        self.weights = weights
        self.tol = float(tol)

        if self.A.ndim != 3:
            raise ValueError("A must have shape [K,N,M]")
        self.K, self.N, self.M = self.A.shape
        if self.B.shape != (self.K, self.N):
            raise ValueError("B must have shape [K,N]")
        if self.Q.shape != self.A.shape or self.X.shape != self.A.shape:
            raise ValueError("Q/X must have shape [K,N,M]")
        if self.distance_metric.shape != (self.N, self.M):
            raise ValueError("distance_metric must have shape [N,M]")
        if self.U.ndim != 3 or self.U.shape[1:] != (self.K, self.M):
            raise ValueError("U must have shape [L,K,M]")
        if self.scheme_idx.shape != (self.M,):
            raise ValueError("scheme_idx must have shape [M]")

        self.selected_demand = self._selected_demand(self.scheme_idx)
        self.total_required = max(
            float(np.sum(np.maximum(self.selected_demand, 0.0))),
            1e-12,
        )

        self.inventory_used = self.A.sum(axis=2).astype(np.int64)
        self.remaining_inventory = self.B - self.inventory_used

        survival = np.maximum(1.0 - self.Q, 0.0)
        self.effective_arrival = np.sum(
            self.A.astype(np.float64) * survival,
            axis=1,
        )
        self.shortage = np.maximum(
            self.selected_demand - self.effective_arrival,
            0.0,
        )

        self.route_load = self.A.sum(axis=0).astype(np.int64)
        route_positions = np.argwhere(self.route_load > 0)
        self.active_routes: set[Route] = {
            (int(i), int(m)) for i, m in route_positions.tolist()
        }
        for route in self.active_routes:
            if route not in self.conflict_index.registered_routes:
                self.conflict_index.register_route(route)
        self.cross_count = self.conflict_index.crossing_count(self.active_routes)

        self.ammo_count = int(np.sum(self.A))
        self.distance_sum = float(
            np.sum(self.A.astype(np.float64) * self.distance_metric[None, :, :])
        )
        self.interception_sum = float(
            np.sum(self.A.astype(np.float64) * self.Q)
        )

        if np.any(self.remaining_inventory < 0):
            raise ValueError("initial A violates inventory")
        if np.any((self.A > 0) & (self.X <= 0.5)):
            raise ValueError("initial A violates reachability")
        if np.any(self.A < 0):
            raise ValueError("initial A must be non-negative")

    def _selected_demand(self, scheme_idx: np.ndarray) -> np.ndarray:
        demand = np.zeros((self.K, self.M), dtype=np.float64)
        for m in range(self.M):
            demand[:, m] = self.U[int(scheme_idx[m]), :, m]
        return demand

    @property
    def cross_base(self) -> float:
        if self.M <= 20:
            denominator = 4.0
        elif self.M <= 40:
            denominator = 6.0
        else:
            denominator = 8.0
        return max((self.M * self.M) / denominator, 1.0)

    def hard_feasible_internal(self) -> bool:
        return bool(
            np.all(self.A >= 0)
            and np.all(self.remaining_inventory >= 0)
            and np.all(self.shortage <= self.tol)
            and np.all((self.A <= 0) | (self.X > 0.5))
        )

    def tactical_components(self) -> dict[str, float]:
        ammo = float(self.ammo_count)
        safe_ammo = max(ammo, 1e-12)
        return {
            "ammo": ammo,
            "ammo_norm": ammo / self.total_required,
            "crossing_count": float(self.cross_count),
            "crossing_norm": float(self.cross_count / self.cross_base),
            "avg_distance": self.distance_sum / safe_ammo if ammo > self.tol else 0.0,
            "avg_interception": (
                self.interception_sum / safe_ammo if ammo > self.tol else 0.0
            ),
        }

    def tactical_cost(self) -> float:
        c = self.tactical_components()
        return float(
            self.weights.ammo * c["ammo_norm"]
            + self.weights.crossing * c["crossing_norm"]
            + self.weights.distance * c["avg_distance"]
            + self.weights.interception * c["avg_interception"]
        )

    def can_add(self, k: int, i: int, m: int, shots: int = 1) -> bool:
        shots = int(shots)
        if shots <= 0:
            return False
        return bool(
            self.remaining_inventory[k, i] >= shots
            and self.X[k, i, m] > 0.5
            and (1.0 - self.Q[k, i, m]) > self.tol
        )

    def delta_cross_if_open(self, i: int, m: int) -> int:
        route = (int(i), int(m))
        return self.conflict_index.query_delta_open(route, self.active_routes)

    def add_shots(self, k: int, i: int, m: int, shots: int) -> None:
        shots = int(shots)
        if not self.can_add(k, i, m, shots):
            raise ValueError("cannot add requested shots")

        route = (int(i), int(m))
        opens = self.route_load[i, m] == 0
        if opens:
            delta_cross = self.conflict_index.query_delta_open(
                route,
                self.active_routes,
            )
            self.conflict_index.register_route(route)
            self.cross_count += int(delta_cross)
            self.active_routes.add(route)

        self.A[k, i, m] += shots
        self.inventory_used[k, i] += shots
        self.remaining_inventory[k, i] -= shots
        self.route_load[i, m] += shots

        survival = max(1.0 - float(self.Q[k, i, m]), 0.0)
        self.effective_arrival[k, m] += shots * survival
        self.shortage[k, m] = max(
            self.selected_demand[k, m] - self.effective_arrival[k, m],
            0.0,
        )

        self.ammo_count += shots
        self.distance_sum += shots * float(self.distance_metric[i, m])
        self.interception_sum += shots * float(self.Q[k, i, m])

    def add_shot(self, k: int, i: int, m: int) -> None:
        self.add_shots(k, i, m, 1)

    def can_remove(self, k: int, i: int, m: int, shots: int = 1) -> bool:
        shots = int(shots)
        if shots <= 0 or self.A[k, i, m] < shots:
            return False
        survival = max(1.0 - float(self.Q[k, i, m]), 0.0)
        after = self.effective_arrival[k, m] - shots * survival
        return bool(after + self.tol >= self.selected_demand[k, m])

    def remove_shots(self, k: int, i: int, m: int, shots: int) -> None:
        shots = int(shots)
        if not self.can_remove(k, i, m, shots):
            raise ValueError("shot removal would violate demand")

        route = (int(i), int(m))
        closes = self.route_load[i, m] == shots
        if closes:
            self.cross_count += self.conflict_index.delta_close(
                route,
                self.active_routes,
            )
            self.active_routes.remove(route)

        self.A[k, i, m] -= shots
        self.inventory_used[k, i] -= shots
        self.remaining_inventory[k, i] += shots
        self.route_load[i, m] -= shots

        survival = max(1.0 - float(self.Q[k, i, m]), 0.0)
        self.effective_arrival[k, m] -= shots * survival
        self.shortage[k, m] = max(
            self.selected_demand[k, m] - self.effective_arrival[k, m],
            0.0,
        )

        self.ammo_count -= shots
        self.distance_sum -= shots * float(self.distance_metric[i, m])
        self.interception_sum -= shots * float(self.Q[k, i, m])

    def remove_shot(self, k: int, i: int, m: int) -> None:
        self.remove_shots(k, i, m, 1)

    def bulk_transfer_effects(
        self,
        k: int,
        old_i: int,
        new_i: int,
        m: int,
        shots: int,
    ) -> dict[str, float | int | bool] | None:
        """Exact incremental effects for a same-target weapon transfer.

        Ammo count is unchanged. Returned average-distance/interception deltas
        therefore equal the corresponding sum deltas divided by current ammo.
        This richer interface lets DeepWTA-Standard enforce topology/distance
        acceptance without re-scanning the full tensor.
        """
        shots = int(shots)
        old_i = int(old_i)
        new_i = int(new_i)
        m = int(m)
        k = int(k)
        if shots <= 0 or old_i == new_i:
            return None
        if self.A[k, old_i, m] < shots:
            return None
        if self.remaining_inventory[k, new_i] < shots:
            return None
        if self.X[k, new_i, m] <= 0.5:
            return None

        old_survival = max(1.0 - float(self.Q[k, old_i, m]), 0.0)
        new_survival = max(1.0 - float(self.Q[k, new_i, m]), 0.0)
        after_arrival = (
            self.effective_arrival[k, m]
            - shots * old_survival
            + shots * new_survival
        )
        if after_arrival + self.tol < self.selected_demand[k, m]:
            return None

        old_route = (old_i, m)
        new_route = (new_i, m)
        old_closes = self.route_load[old_i, m] == shots
        new_opens = self.route_load[new_i, m] == 0

        delta_cross = 0
        active_after_old = self.active_routes
        if old_closes:
            delta_cross += self.conflict_index.delta_close(
                old_route,
                active_after_old,
            )
            active_after_old = active_after_old - {old_route}
        if new_opens:
            delta_cross += self.conflict_index.query_delta_open(
                new_route,
                active_after_old,
            )

        ammo = max(float(self.ammo_count), 1e-12)
        delta_distance_sum = shots * (
            float(self.distance_metric[new_i, m])
            - float(self.distance_metric[old_i, m])
        )
        delta_interception_sum = shots * (
            float(self.Q[k, new_i, m])
            - float(self.Q[k, old_i, m])
        )
        delta_avg_distance = delta_distance_sum / ammo
        delta_avg_interception = delta_interception_sum / ammo
        delta_cost = (
            self.weights.crossing * (delta_cross / self.cross_base)
            + self.weights.distance * delta_avg_distance
            + self.weights.interception * delta_avg_interception
        )

        return {
            "delta_cost": float(delta_cost),
            "delta_cross": int(delta_cross),
            "delta_distance_sum": float(delta_distance_sum),
            "delta_avg_distance": float(delta_avg_distance),
            "delta_interception_sum": float(delta_interception_sum),
            "delta_avg_interception": float(delta_avg_interception),
            "projected_avg_distance": float(
                (self.distance_sum + delta_distance_sum) / ammo
            ),
            "projected_crossing_count": int(self.cross_count + delta_cross),
            "old_route_closes": bool(old_closes),
            "new_route_opens": bool(new_opens),
        }

    def bulk_transfer_delta(
        self,
        k: int,
        old_i: int,
        new_i: int,
        m: int,
        shots: int,
    ) -> float | None:
        effects = self.bulk_transfer_effects(k, old_i, new_i, m, shots)
        if effects is None:
            return None
        return float(effects["delta_cost"])

    def apply_bulk_transfer(
        self,
        k: int,
        old_i: int,
        new_i: int,
        m: int,
        shots: int,
    ) -> None:
        delta = self.bulk_transfer_delta(k, old_i, new_i, m, shots)
        if delta is None:
            raise ValueError("invalid bulk transfer")

        shots = int(shots)
        old_route = (int(old_i), int(m))
        new_route = (int(new_i), int(m))
        old_closes = self.route_load[old_i, m] == shots
        new_opens = self.route_load[new_i, m] == 0

        if old_closes:
            self.cross_count += self.conflict_index.delta_close(
                old_route,
                self.active_routes,
            )
            self.active_routes.remove(old_route)

        if new_opens:
            delta_open = self.conflict_index.query_delta_open(
                new_route,
                self.active_routes,
            )
            self.conflict_index.register_route(new_route)
            self.cross_count += delta_open
            self.active_routes.add(new_route)

        self.A[k, old_i, m] -= shots
        self.A[k, new_i, m] += shots
        self.inventory_used[k, old_i] -= shots
        self.inventory_used[k, new_i] += shots
        self.remaining_inventory[k, old_i] += shots
        self.remaining_inventory[k, new_i] -= shots
        self.route_load[old_i, m] -= shots
        self.route_load[new_i, m] += shots

        old_survival = max(1.0 - float(self.Q[k, old_i, m]), 0.0)
        new_survival = max(1.0 - float(self.Q[k, new_i, m]), 0.0)
        self.effective_arrival[k, m] += shots * (new_survival - old_survival)
        self.shortage[k, m] = max(
            self.selected_demand[k, m] - self.effective_arrival[k, m],
            0.0,
        )

        self.distance_sum += shots * (
            float(self.distance_metric[new_i, m])
            - float(self.distance_metric[old_i, m])
        )
        self.interception_sum += shots * (
            float(self.Q[k, new_i, m])
            - float(self.Q[k, old_i, m])
        )

    def replacement_effects(
        self,
        k: int,
        old_i: int,
        new_i: int,
        m: int,
        remove_shots: int,
        add_shots: int,
    ):
        """Exact effects of replacing shots on one route by another route.

        Unlike ``bulk_transfer_effects``, the replacement may change the total
        ammunition count.  This is the primitive used by archived resource refinement.
        """
        k = int(k)
        old_i = int(old_i)
        new_i = int(new_i)
        m = int(m)
        remove_shots = int(remove_shots)
        add_shots = int(add_shots)

        if old_i == new_i or remove_shots <= 0 or add_shots < 0:
            return None
        if self.A[k, old_i, m] < remove_shots:
            return None
        if add_shots > 0:
            if self.X[k, new_i, m] <= 0.5:
                return None
            if self.remaining_inventory[k, new_i] < add_shots:
                return None
            new_survival = max(1.0 - float(self.Q[k, new_i, m]), 0.0)
            if new_survival <= self.tol:
                return None
        else:
            new_survival = 0.0

        old_survival = max(1.0 - float(self.Q[k, old_i, m]), 0.0)
        after_arrival = (
            self.effective_arrival[k, m]
            - remove_shots * old_survival
            + add_shots * new_survival
        )
        if after_arrival + self.tol < self.selected_demand[k, m]:
            return None

        old_route = (old_i, m)
        new_route = (new_i, m)
        old_closes = self.route_load[old_i, m] == remove_shots
        new_opens = add_shots > 0 and self.route_load[new_i, m] == 0

        delta_cross = 0
        active_after_old = self.active_routes
        if old_closes:
            delta_cross += self.conflict_index.delta_close(
                old_route,
                active_after_old,
            )
            active_after_old = active_after_old - {old_route}
        if new_opens:
            delta_cross += self.conflict_index.query_delta_open(
                new_route,
                active_after_old,
            )

        delta_ammo = int(add_shots - remove_shots)
        ammo_after = int(self.ammo_count + delta_ammo)
        if ammo_after <= 0:
            return None

        delta_distance_sum = (
            add_shots * float(self.distance_metric[new_i, m])
            - remove_shots * float(self.distance_metric[old_i, m])
        )
        delta_interception_sum = (
            add_shots * float(self.Q[k, new_i, m])
            - remove_shots * float(self.Q[k, old_i, m])
        )

        current = self.tactical_components()
        projected_avg_distance = (
            self.distance_sum + delta_distance_sum
        ) / float(ammo_after)
        projected_avg_interception = (
            self.interception_sum + delta_interception_sum
        ) / float(ammo_after)
        projected_ammo_norm = float(ammo_after) / self.total_required
        projected_crossing_count = int(self.cross_count + delta_cross)
        projected_crossing_norm = projected_crossing_count / self.cross_base

        projected_cost = (
            self.weights.ammo * projected_ammo_norm
            + self.weights.crossing * projected_crossing_norm
            + self.weights.distance * projected_avg_distance
            + self.weights.interception * projected_avg_interception
        )

        return {
            "delta_cost": float(projected_cost - self.tactical_cost()),
            "delta_ammo": int(delta_ammo),
            "ammo_saved": int(remove_shots - add_shots),
            "delta_cross": int(delta_cross),
            "delta_distance_sum": float(delta_distance_sum),
            "delta_interception_sum": float(delta_interception_sum),
            "projected_ammo_norm": float(projected_ammo_norm),
            "projected_avg_distance": float(projected_avg_distance),
            "projected_avg_interception": float(projected_avg_interception),
            "distance_improvement": float(
                current["avg_distance"] - projected_avg_distance
            ),
            "interception_improvement": float(
                current["avg_interception"] - projected_avg_interception
            ),
            "projected_crossing_count": int(projected_crossing_count),
            "old_route_closes": bool(old_closes),
            "new_route_opens": bool(new_opens),
        }

    def apply_replacement(
        self,
        k: int,
        old_i: int,
        new_i: int,
        m: int,
        remove_shots: int,
        add_shots: int,
    ) -> None:
        effects = self.replacement_effects(
            k,
            old_i,
            new_i,
            m,
            remove_shots,
            add_shots,
        )
        if effects is None:
            raise ValueError("invalid resource replacement")

        k = int(k)
        old_i = int(old_i)
        new_i = int(new_i)
        m = int(m)
        remove_shots = int(remove_shots)
        add_shots = int(add_shots)
        old_route = (old_i, m)
        new_route = (new_i, m)

        old_closes = self.route_load[old_i, m] == remove_shots
        if old_closes:
            self.cross_count += self.conflict_index.delta_close(
                old_route,
                self.active_routes,
            )
            self.active_routes.remove(old_route)

        new_opens = add_shots > 0 and self.route_load[new_i, m] == 0
        if new_opens:
            delta_open = self.conflict_index.query_delta_open(
                new_route,
                self.active_routes,
            )
            self.conflict_index.register_route(new_route)
            self.cross_count += int(delta_open)
            self.active_routes.add(new_route)

        self.A[k, old_i, m] -= remove_shots
        self.inventory_used[k, old_i] -= remove_shots
        self.remaining_inventory[k, old_i] += remove_shots
        self.route_load[old_i, m] -= remove_shots

        if add_shots > 0:
            self.A[k, new_i, m] += add_shots
            self.inventory_used[k, new_i] += add_shots
            self.remaining_inventory[k, new_i] -= add_shots
            self.route_load[new_i, m] += add_shots

        old_survival = max(1.0 - float(self.Q[k, old_i, m]), 0.0)
        new_survival = (
            max(1.0 - float(self.Q[k, new_i, m]), 0.0)
            if add_shots > 0
            else 0.0
        )
        self.effective_arrival[k, m] += (
            add_shots * new_survival - remove_shots * old_survival
        )
        self.shortage[k, m] = max(
            self.selected_demand[k, m] - self.effective_arrival[k, m],
            0.0,
        )

        self.ammo_count += add_shots - remove_shots
        self.distance_sum += (
            add_shots * float(self.distance_metric[new_i, m])
            - remove_shots * float(self.distance_metric[old_i, m])
        )
        self.interception_sum += (
            add_shots * float(self.Q[k, new_i, m])
            - remove_shots * float(self.Q[k, old_i, m])
        )

    def route_transfer_effects(
        self,
        old_i: int,
        new_i: int,
        m: int,
    ) -> dict[str, float | int | bool] | None:
        """Exact effects for moving an ENTIRE active route.

        This is the main DeepWTA-Standard primitive: it changes route topology
        while preserving total ammunition. The move is returned only when hard
        inventory/reachability/effective-arrival feasibility is preserved.
        """
        old_i = int(old_i)
        new_i = int(new_i)
        m = int(m)
        if old_i == new_i or self.route_load[old_i, m] <= 0:
            return None

        shots = self.A[:, old_i, m].astype(np.int64)
        positive = shots > 0
        if not np.any(positive):
            return None
        if np.any(self.X[positive, new_i, m] <= 0.5):
            return None
        if np.any(self.remaining_inventory[positive, new_i] < shots[positive]):
            return None

        old_survival = np.maximum(1.0 - self.Q[:, old_i, m], 0.0)
        new_survival = np.maximum(1.0 - self.Q[:, new_i, m], 0.0)
        after_arrival = self.effective_arrival[:, m] + shots * (
            new_survival - old_survival
        )
        if np.any(after_arrival + self.tol < self.selected_demand[:, m]):
            return None

        old_route = (old_i, m)
        new_route = (new_i, m)
        delta_cross = self.conflict_index.delta_close(
            old_route,
            self.active_routes,
        )
        active_after_old = self.active_routes - {old_route}
        new_opens = new_route not in active_after_old
        if new_opens:
            delta_cross += self.conflict_index.query_delta_open(
                new_route,
                active_after_old,
            )

        total_shots = int(np.sum(shots))
        ammo = max(float(self.ammo_count), 1e-12)
        delta_distance_sum = total_shots * (
            float(self.distance_metric[new_i, m])
            - float(self.distance_metric[old_i, m])
        )
        delta_interception_sum = float(
            np.sum(shots * (self.Q[:, new_i, m] - self.Q[:, old_i, m]))
        )
        delta_avg_distance = delta_distance_sum / ammo
        delta_avg_interception = delta_interception_sum / ammo
        delta_cost = (
            self.weights.crossing * (delta_cross / self.cross_base)
            + self.weights.distance * delta_avg_distance
            + self.weights.interception * delta_avg_interception
        )

        return {
            "delta_cost": float(delta_cost),
            "delta_cross": int(delta_cross),
            "delta_distance_sum": float(delta_distance_sum),
            "delta_avg_distance": float(delta_avg_distance),
            "delta_interception_sum": float(delta_interception_sum),
            "delta_avg_interception": float(delta_avg_interception),
            "projected_avg_distance": float(
                (self.distance_sum + delta_distance_sum) / ammo
            ),
            "projected_crossing_count": int(self.cross_count + delta_cross),
            "old_route_closes": True,
            "new_route_opens": bool(new_opens),
            "moved_shots": int(total_shots),
        }

    def route_transfer_delta(
        self,
        old_i: int,
        new_i: int,
        m: int,
    ) -> float | None:
        effects = self.route_transfer_effects(old_i, new_i, m)
        if effects is None:
            return None
        return float(effects["delta_cost"])

    def apply_route_transfer(self, old_i: int, new_i: int, m: int) -> None:
        delta = self.route_transfer_delta(old_i, new_i, m)
        if delta is None:
            raise ValueError("invalid route-level transfer")

        old_i = int(old_i)
        new_i = int(new_i)
        m = int(m)
        shots = self.A[:, old_i, m].astype(np.int64).copy()
        old_route = (old_i, m)
        new_route = (new_i, m)

        self.cross_count += self.conflict_index.delta_close(
            old_route,
            self.active_routes,
        )
        self.active_routes.remove(old_route)

        new_opens = self.route_load[new_i, m] == 0
        if new_opens:
            delta_open = self.conflict_index.query_delta_open(
                new_route,
                self.active_routes,
            )
            self.conflict_index.register_route(new_route)
            self.cross_count += delta_open
            self.active_routes.add(new_route)

        self.A[:, old_i, m] = 0
        self.A[:, new_i, m] += shots
        self.inventory_used[:, old_i] -= shots
        self.inventory_used[:, new_i] += shots
        self.remaining_inventory[:, old_i] += shots
        self.remaining_inventory[:, new_i] -= shots
        self.route_load[old_i, m] = 0
        self.route_load[new_i, m] += int(np.sum(shots))

        old_survival = np.maximum(1.0 - self.Q[:, old_i, m], 0.0)
        new_survival = np.maximum(1.0 - self.Q[:, new_i, m], 0.0)
        self.effective_arrival[:, m] += shots * (new_survival - old_survival)
        self.shortage[:, m] = np.maximum(
            self.selected_demand[:, m] - self.effective_arrival[:, m],
            0.0,
        )

        total_shots = int(np.sum(shots))
        self.distance_sum += total_shots * (
            float(self.distance_metric[new_i, m])
            - float(self.distance_metric[old_i, m])
        )
        self.interception_sum += float(
            np.sum(shots * (self.Q[:, new_i, m] - self.Q[:, old_i, m]))
        )

    def force_scheme(self, target: int, new_scheme: int) -> None:
        target = int(target)
        new_scheme = int(new_scheme)
        new_demand = self.U[new_scheme, :, target]
        old_total = float(np.sum(self.selected_demand[:, target]))
        new_total = float(np.sum(new_demand))
        self.scheme_idx[target] = new_scheme
        self.selected_demand[:, target] = new_demand
        self.total_required = max(
            self.total_required - old_total + new_total,
            1e-12,
        )
        self.shortage[:, target] = np.maximum(
            new_demand - self.effective_arrival[:, target],
            0.0,
        )

    def set_scheme_if_already_feasible(self, target: int, new_scheme: int) -> bool:
        target = int(target)
        new_scheme = int(new_scheme)
        new_demand = self.U[new_scheme, :, target]
        if np.any(self.effective_arrival[:, target] + self.tol < new_demand):
            return False
        self.force_scheme(target, new_scheme)
        return True

    def clone(self) -> "IncrementalRepairState":
        clone = object.__new__(IncrementalRepairState)
        clone.A = self.A.copy()
        clone.scheme_idx = self.scheme_idx.copy()
        clone.B = self.B
        clone.U = self.U
        clone.Q = self.Q
        clone.X = self.X
        clone.distance_metric = self.distance_metric
        clone.conflict_index = self.conflict_index
        clone.weights = self.weights
        clone.tol = self.tol
        clone.K = self.K
        clone.N = self.N
        clone.M = self.M
        clone.selected_demand = self.selected_demand.copy()
        clone.total_required = float(self.total_required)
        clone.inventory_used = self.inventory_used.copy()
        clone.remaining_inventory = self.remaining_inventory.copy()
        clone.effective_arrival = self.effective_arrival.copy()
        clone.shortage = self.shortage.copy()
        clone.route_load = self.route_load.copy()
        clone.active_routes = set(self.active_routes)
        clone.cross_count = int(self.cross_count)
        clone.ammo_count = int(self.ammo_count)
        clone.distance_sum = float(self.distance_sum)
        clone.interception_sum = float(self.interception_sum)
        return clone

    def replace_from(self, other: "IncrementalRepairState") -> None:
        self.A = other.A.copy()
        self.scheme_idx = other.scheme_idx.copy()
        self.selected_demand = other.selected_demand.copy()
        self.total_required = float(other.total_required)
        self.inventory_used = other.inventory_used.copy()
        self.remaining_inventory = other.remaining_inventory.copy()
        self.effective_arrival = other.effective_arrival.copy()
        self.shortage = other.shortage.copy()
        self.route_load = other.route_load.copy()
        self.active_routes = set(other.active_routes)
        self.cross_count = int(other.cross_count)
        self.ammo_count = int(other.ammo_count)
        self.distance_sum = float(other.distance_sum)
        self.interception_sum = float(other.interception_sum)
