from __future__ import annotations

from typing import Iterable

import numpy as np

Route = tuple[int, int]
CrossPair = tuple[Route, Route]


def _point_xy(point: np.ndarray) -> tuple[float, float]:
    point = np.asarray(point, dtype=float).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        raise ValueError("each coordinate must contain two finite values")
    return float(point[1]), float(point[0])


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ax, ay = _point_xy(a)
    bx, by = _point_xy(b)
    cx, cy = _point_xy(c)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _sign(value: float, eps: float) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _positive_collinear_overlap(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    eps: float,
) -> bool:
    ax, ay = _point_xy(a)
    bx, by = _point_xy(b)
    cx, cy = _point_xy(c)
    dx, dy = _point_xy(d)

    if abs(bx - ax) >= abs(by - ay):
        left = max(min(ax, bx), min(cx, dx))
        right = min(max(ax, bx), max(cx, dx))
    else:
        left = max(min(ay, by), min(cy, dy))
        right = min(max(ay, by), max(cy, dy))

    return right - left > eps


def segments_conflict(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    eps: float = 1e-10,
) -> bool:
    """Return True for an interior crossing or positive collinear overlap."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    s1 = _sign(o1, eps)
    s2 = _sign(o2, eps)
    s3 = _sign(o3, eps)
    s4 = _sign(o4, eps)

    if s1 * s2 < 0 and s3 * s4 < 0:
        return True

    if s1 == s2 == s3 == s4 == 0:
        return _positive_collinear_overlap(a, b, c, d, eps)

    return False


def route_conflict(
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    route_a: Route,
    route_b: Route,
    eps: float = 1e-10,
) -> bool:
    """Check two base-target routes; shared endpoints are not conflicts."""
    base_coords = np.asarray(base_coords, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)
    n1, m1 = route_a
    n2, m2 = route_b

    if route_a == route_b or n1 == n2 or m1 == m2:
        return False

    return segments_conflict(
        base_coords[n1],
        target_coords[m1],
        base_coords[n2],
        target_coords[m2],
        eps=eps,
    )


def active_routes(allocations: np.ndarray, tol: float = 1e-7) -> list[Route]:
    allocations = np.asarray(allocations, dtype=float)
    if allocations.ndim != 3:
        raise ValueError("allocations must have shape [K, N, M]")

    route_load = np.sum(np.maximum(allocations, 0.0), axis=0)
    return [tuple(map(int, item)) for item in np.argwhere(route_load > tol)]


def find_crossing_pairs(
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    routes: Iterable[Route],
    eps: float = 1e-10,
) -> list[CrossPair]:
    route_list = sorted(set(tuple(map(int, route)) for route in routes))
    if len(route_list) < 2:
        return []

    base_coords = np.asarray(base_coords, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)
    route_array = np.asarray(route_list, dtype=np.int64)

    launch = base_coords[route_array[:, 0], :2]
    target = target_coords[route_array[:, 1], :2]
    px, py = launch[:, 1], launch[:, 0]
    qx, qy = target[:, 1], target[:, 0]

    pairs: list[CrossPair] = []
    for i in range(len(route_list) - 1):
        js = np.arange(i + 1, len(route_list))
        distinct = (route_array[js, 0] != route_array[i, 0]) & (
            route_array[js, 1] != route_array[i, 1]
        )
        if not np.any(distinct):
            continue
        js = js[distinct]

        ax, ay = px[i], py[i]
        bx, by = qx[i], qy[i]
        cx, cy = px[js], py[js]
        dx, dy = qx[js], qy[js]

        o1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        o2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
        o3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
        o4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)

        proper = (((o1 > eps) & (o2 < -eps)) | ((o1 < -eps) & (o2 > eps))) & (
            ((o3 > eps) & (o4 < -eps)) | ((o3 < -eps) & (o4 > eps))
        )

        collinear = (
            (np.abs(o1) <= eps)
            & (np.abs(o2) <= eps)
            & (np.abs(o3) <= eps)
            & (np.abs(o4) <= eps)
        )
        overlap = np.zeros(len(js), dtype=bool)
        if np.any(collinear):
            col_js = js[collinear]
            if abs(bx - ax) >= abs(by - ay):
                left = np.maximum(min(ax, bx), np.minimum(px[col_js], qx[col_js]))
                right = np.minimum(max(ax, bx), np.maximum(px[col_js], qx[col_js]))
            else:
                left = np.maximum(min(ay, by), np.minimum(py[col_js], qy[col_js]))
                right = np.minimum(max(ay, by), np.maximum(py[col_js], qy[col_js]))
            overlap[collinear] = right - left > eps

        for j in js[proper | (collinear & overlap)]:
            pairs.append((route_list[i], route_list[int(j)]))

    return pairs


def crossing_pairs_from_allocations(
    allocations: np.ndarray,
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    tol: float = 1e-7,
    eps: float = 1e-10,
) -> list[CrossPair]:
    return find_crossing_pairs(
        base_coords,
        target_coords,
        active_routes(allocations, tol=tol),
        eps=eps,
    )


def count_crossings(
    allocations: np.ndarray,
    base_coords: np.ndarray,
    target_coords: np.ndarray,
    tol: float = 1e-7,
    eps: float = 1e-10,
) -> int:
    return len(
        crossing_pairs_from_allocations(
            allocations,
            base_coords,
            target_coords,
            tol=tol,
            eps=eps,
        )
    )
