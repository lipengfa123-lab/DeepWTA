#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    root = args.release_root.resolve()
    out = (args.output or root / "manifests" / "SHA256_MANIFEST.csv").resolve()
    skip_dir = out.parent if out.parent != root else None
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == out:
            continue
        if skip_dir is not None and skip_dir in path.parents:
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha(path),
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[manifest] {len(rows)} files -> {out}")


if __name__ == "__main__":
    main()
