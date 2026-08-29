from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from deepwta.utils.geometry import find_crossing_pairs, route_conflict


Route = tuple[int, int]


def _pair_key(a: Route, b: Route) -> tuple[Route, Route]:
    return (a, b) if a <= b else (b, a)


class LazyConflictIndex:
    """Lazy exact route-conflict index for tactical refinement.

    Important design choice:
      * Only routes that have actually been active are registered permanently.
      * Candidate routes are queried only against the CURRENT active set.
      * Exact geometry results are cached pairwise.

    This avoids the historical eager-index failure mode precomputed a conflict graph for
    every Top-K candidate route and became much slower than the lazy index on large/xlarge
    scenarios.
    """

    def __init__(
        self,
        base_coords: np.ndarray,
        target_coords: np.ndarray,
        initial_active_routes: Iterable[Route] = (),
    ):
        self.base_coords = np.asarray(base_coords, dtype=np.float64)
        self.target_coords = np.asarray(target_coords, dtype=np.float64)

        self.registered_routes: set[Route] = set()
        self.adj: dict[Route, set[Route]] = defaultdict(set)
        self.pair_cache: dict[tuple[Route, Route], bool] = {}

        self.geometry_tests = 0
        self.cache_hits = 0
        self.query_calls = 0
        self.register_calls = 0
        self.initial_route_count = 0
        self._initialize_active(initial_active_routes)

    def _initialize_active(self, routes: Iterable[Route]) -> None:
        route_list = sorted(set(tuple(map(int, route)) for route in routes))
        self.initial_route_count = len(route_list)
        self.registered_routes.update(route_list)
        for route in route_list:
            self.adj.setdefault(route, set())

        if len(route_list) < 2:
            return

        # Reuse the project's exact broad-phase implementation for the initial
        # active set. Only positive pairs need adjacency entries; subsequent new
        # routes will be checked lazily and cached.
        for a, b in find_crossing_pairs(
            self.base_coords,
            self.target_coords,
            route_list,
        ):
            a = tuple(map(int, a))
            b = tuple(map(int, b))
            self.adj[a].add(b)
            self.adj[b].add(a)
            self.pair_cache[_pair_key(a, b)] = True

    def _conflict(self, route_a: Route, route_b: Route) -> bool:
        a = tuple(map(int, route_a))
        b = tuple(map(int, route_b))
        if a == b:
            return False

        key = _pair_key(a, b)
        cached = self.pair_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return bool(cached)

        self.geometry_tests += 1
        value = bool(
            route_conflict(
                self.base_coords,
                self.target_coords,
                a,
                b,
            )
        )
        self.pair_cache[key] = value
        return value

    def query_delta_open(
        self,
        route: Route,
        active_routes: set[Route],
    ) -> int:
        """Exact crossing increment if route were opened now.

        Querying does NOT register the route. This is the key lazy behavior: a
        route that is merely scored during candidate search never forces full
        adjacency construction against every other candidate route.
        """
        route = tuple(map(int, route))
        self.query_calls += 1
        if route in active_routes:
            return 0

        if route in self.registered_routes:
            return int(
                sum(1 for other in self.adj[route] if other in active_routes)
            )

        count = 0
        for other in active_routes:
            if self._conflict(route, other):
                count += 1
        return int(count)

    def register_route(self, route: Route) -> None:
        """Register a route that actually becomes active.

        Adjacency is completed only against routes that have themselves been
        active at least once. Candidate-only routes are never included.
        """
        route = tuple(map(int, route))
        if route in self.registered_routes:
            return

        self.register_calls += 1
        existing = list(self.registered_routes)
        self.registered_routes.add(route)
        self.adj.setdefault(route, set())

        for other in existing:
            if self._conflict(route, other):
                self.adj[route].add(other)
                self.adj[other].add(route)

    def delta_close(
        self,
        route: Route,
        active_routes: set[Route],
    ) -> int:
        route = tuple(map(int, route))
        if route not in active_routes:
            return 0
        if route not in self.registered_routes:
            raise RuntimeError("active route is missing from LazyConflictIndex")
        return -int(
            sum(
                1
                for other in self.adj[route]
                if other in active_routes and other != route
            )
        )

    def crossing_count(self, active_routes: set[Route]) -> int:
        count = 0
        for route in active_routes:
            if route not in self.registered_routes:
                raise RuntimeError("active route is missing from LazyConflictIndex")
            for other in self.adj[route]:
                if other in active_routes and route < other:
                    count += 1
        return int(count)

    def target_crossing_degree(
        self,
        active_routes: set[Route],
        num_targets: int,
    ) -> np.ndarray:
        degree = np.zeros(int(num_targets), dtype=np.int64)
        for route in active_routes:
            for other in self.adj.get(route, ()):
                if other not in active_routes or route >= other:
                    continue
                degree[route[1]] += 1
                degree[other[1]] += 1
        return degree

    def route_crossing_degree(
        self,
        route: Route,
        active_routes: set[Route],
    ) -> int:
        route = tuple(map(int, route))
        if route not in active_routes:
            return 0
        return int(
            sum(1 for other in self.adj.get(route, ()) if other in active_routes)
        )

    def summary(self) -> dict[str, float]:
        undirected_edges = sum(len(v) for v in self.adj.values()) // 2
        return {
            "conflict_index_initial_routes": float(self.initial_route_count),
            "conflict_index_registered_routes": float(len(self.registered_routes)),
            "conflict_index_edges": float(undirected_edges),
            "conflict_index_mean_degree": float(
                2 * undirected_edges / max(len(self.registered_routes), 1)
            ),
            "conflict_index_pair_cache_size": float(len(self.pair_cache)),
            "conflict_index_geometry_tests": float(self.geometry_tests),
            "conflict_index_cache_hits": float(self.cache_hits),
            "conflict_index_query_calls": float(self.query_calls),
            "conflict_index_register_calls": float(self.register_calls),
        }
