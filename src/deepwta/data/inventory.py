from __future__ import annotations

import numpy as np


def fit_inventory_to_total(
    used: np.ndarray,
    preferred_inventory: np.ndarray,
    *,
    total_weapons: int,
    inventory_max: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Return B with an exact global sum while preserving A_ref <= B."""
    used = np.asarray(used, dtype=np.int16)
    preferred = np.asarray(preferred_inventory, dtype=np.int16)
    requested = int(total_weapons)
    used_total = int(np.sum(used))
    max_total = int(used.size * inventory_max)
    if requested < used_total or requested > max_total:
        return None

    preferred_extra = np.maximum(preferred - used, 0).astype(np.float64)
    inventory = used.copy()
    remaining = requested - used_total
    flat = inventory.reshape(-1)
    weights = (preferred_extra.reshape(-1) + 1.0).astype(np.float64)
    while remaining > 0:
        available = np.flatnonzero(flat < inventory_max)
        if available.size == 0:
            return None
        probabilities = weights[available]
        probabilities /= probabilities.sum()
        chosen = int(rng.choice(available, p=probabilities))
        flat[chosen] += 1
        remaining -= 1
    return flat.reshape(used.shape).astype(np.int16)
