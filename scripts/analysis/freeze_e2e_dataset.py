from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Hash/freeze a shared E2E dataset before model training.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    project = Path(__file__).resolve().parents[2]
    manifest = args.manifest.resolve()
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    files = []
    for split, paths in meta.get("files", {}).items():
        for raw in paths:
            path = Path(raw)
            if not path.is_absolute():
                path = project / path
            if not path.exists():
                raise FileNotFoundError(path)
            files.append({"split": split, "path": str(path.relative_to(project)), "sha256": sha256(path), "bytes": path.stat().st_size})

    payload = {
        "manifest": str(manifest.relative_to(project) if manifest.is_relative_to(project) else manifest),
        "manifest_sha256": sha256(manifest),
        "dataset_name": meta.get("dataset_name"),
        "version": meta.get("version"),
        "seed": meta.get("seed"),
        "split": meta.get("split"),
        "map_hash": meta.get("map_hash"),
        "files": files,
        "contract": "All subsequent DeepWTA-Standard runs must use these exact bytes.",
    }
    output = args.output or manifest.with_suffix(".freeze.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"freeze file: {output}")
    print(f"manifest sha256: {payload['manifest_sha256']}")
    print(f"shards hashed: {len(files)}")


if __name__ == "__main__":
    main()
