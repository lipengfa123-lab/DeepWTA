from __future__ import annotations

import numpy as np

from deepwta.utils.geometry import route_conflict, segments_conflict


def main():
    # Proper interior crossing.
    assert segments_conflict(
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 0.0]),
    )

    # Endpoint touch only: not a conflict.
    assert not segments_conflict(
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([2.0, 0.0]),
    )

    # Positive collinear overlap: conflict.
    assert segments_conflict(
        np.asarray([0.0, 0.0]),
        np.asarray([2.0, 0.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([3.0, 0.0]),
    )

    # Shared base or shared target is explicitly excluded at route level.
    bases = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=float)
    targets = np.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=float)
    assert not route_conflict(bases, targets, (0, 0), (0, 1))
    assert not route_conflict(bases, targets, (0, 0), (1, 0))

    print("PASS: exact crossing semantics")


if __name__ == "__main__":
    main()
