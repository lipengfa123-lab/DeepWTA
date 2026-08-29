#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deepwta.utils.checkpoint_registry import (
    DEFAULT_REGISTRY_PATH,
    all_registered_checkpoints,
    verify_registered_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify registered checkpoint files and hashes.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    records = all_registered_checkpoints(args.registry)
    for checkpoint in records:
        verify_registered_checkpoint(checkpoint, verify_hash=not args.skip_hash)
        print(
            f"OK {checkpoint.group}/{checkpoint.name}/{checkpoint.artifact} "
            f"{checkpoint.sha256[:16]}..."
        )
    print(f"verified {len(records)} registered checkpoint artifacts")


if __name__ == "__main__":
    main()
