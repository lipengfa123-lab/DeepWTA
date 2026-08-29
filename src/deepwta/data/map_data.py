from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class MapContext:
    height: int
    width: int
    max_distance: float
    base_candidates: np.ndarray  # [P,2], row/y then col/x
    target_candidates: np.ndarray  # [Q,2], row/y then col/x


def load_land_mask(path: str | Path) -> np.ndarray:
    """Return a binary land mask using the same ocean-removal logic as the source project."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read map: {path}")

    blue, green, red = cv2.split(image)
    blue = blue.astype(np.int32)
    green = green.astype(np.int32)
    red = red.astype(np.int32)

    not_black = (blue + green + red) > 30
    not_ocean = (blue - red) < 20
    mask = (not_black & not_ocean).astype(np.uint8) * 255

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def build_grid_candidates(mask: np.ndarray, step: int, margin: int) -> np.ndarray:
    """Build valid grid centers inside a binary mask."""
    if step <= 0:
        raise ValueError("step must be positive")

    height, width = mask.shape[:2]
    if height <= 2 * margin or width <= 2 * margin:
        raise ValueError("map is too small for the configured margin")

    half = step // 2
    y_start = max(margin, half)
    x_start = max(margin, half)

    ys = np.arange(y_start, height - margin, step, dtype=np.int32)
    xs = np.arange(x_start, width - margin, step, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")

    rows = grid_y.reshape(-1)
    cols = grid_x.reshape(-1)
    valid = mask[rows, cols]
    return np.stack([rows[valid], cols[valid]], axis=1).astype(np.float32)


def load_map_context(
    friendly_map: str | Path,
    enemy_map: str | Path,
    friendly_step: int = 8,
    enemy_step: int = 5,
    margin: int = 10,
) -> MapContext:
    friendly_mask = load_land_mask(friendly_map)
    enemy_mask = load_land_mask(enemy_map)
    if friendly_mask.shape != enemy_mask.shape:
        raise ValueError("Friendly_Map and Enemy_Map must have the same image size")

    height, width = friendly_mask.shape
    base_candidates = build_grid_candidates(friendly_mask, friendly_step, margin)
    target_candidates = build_grid_candidates(enemy_mask, enemy_step, margin)

    if len(base_candidates) == 0 or len(target_candidates) == 0:
        raise ValueError("no valid base/target grid candidates were extracted from the maps")

    return MapContext(
        height=height,
        width=width,
        max_distance=math.hypot(height, width),
        base_candidates=base_candidates,
        target_candidates=target_candidates,
    )


def sample_coordinates(
    candidates: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count > len(candidates):
        raise ValueError(f"requested {count} points but only {len(candidates)} candidates exist")
    indices = rng.choice(len(candidates), size=count, replace=False)
    return candidates[indices].copy()


def pairwise_distance(base_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    delta = base_coords[:, None, :] - target_coords[None, :, :]
    return np.linalg.norm(delta, axis=-1).astype(np.float32)
