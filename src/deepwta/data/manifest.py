from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from deepwta.utils.paths import PROJECT_ROOT, project_path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    dataset_name: str
    scale: str | None
    seed: int
    root: Path
    payload: dict[str, Any]


def _portable_path(value: str | Path) -> str | Path:
    """Normalize manifest paths written on another OS.

    Historical dataset manifests in this project were generated on Windows and
    therefore contain values such as ``data\\processed\\wta_small``.  On Linux,
    ``Path`` treats the backslashes as literal characters.  Convert separators
    before resolving project-relative paths.  Absolute POSIX paths are left
    untouched, and this conversion is harmless on Windows as well.
    """
    if isinstance(value, Path):
        value = str(value)
    text = str(value)
    if os.sep == "/":
        text = text.replace("\\", "/")
    else:
        text = text.replace("/", "\\")
    return text


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = project_path(_portable_path(path))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_name = str(payload.get("dataset_name") or "").strip()
    if not dataset_name:
        raise ValueError(f"manifest has no dataset_name: {manifest_path}")
    root_value = payload.get("root")
    if not root_value:
        raise ValueError(f"manifest has no dataset root: {manifest_path}")
    root = project_path(_portable_path(root_value))
    if not root.is_dir():
        raise FileNotFoundError(f"manifest dataset root does not exist: {root}")
    return DatasetManifest(
        path=manifest_path,
        dataset_name=dataset_name,
        scale=payload.get("scale"),
        seed=int(payload.get("seed", payload.get("split", {}).get("base_seed", -1))),
        root=root,
        payload=payload,
    )


def relative_to_project(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
