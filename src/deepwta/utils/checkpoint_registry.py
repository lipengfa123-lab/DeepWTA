from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from deepwta.utils.paths import CONFIGS_DIR, project_path


DEFAULT_REGISTRY_PATH = CONFIGS_DIR / "registry" / "checkpoints.json"
DEEPWTA_PROFILE_ALIASES = {
    "standard": "standard",
}


@dataclass(frozen=True)
class RegisteredCheckpoint:
    group: str
    name: str
    display_name: str
    artifact: str
    path: Path
    sha256: str
    bytes: int
    metadata: dict[str, Any]


def load_checkpoint_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = project_path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported checkpoint registry schema: {registry_path}")
    return payload


def _resolve(
    group: str,
    name: str,
    artifact: str,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> RegisteredCheckpoint:
    registry = load_checkpoint_registry(registry_path)
    if group == "deepwta":
        entries = registry["deepwta"]["profiles"]
    else:
        entries = registry[group]
    try:
        entry = entries[name]
        artifact_entry = entry["artifacts"][artifact]
    except KeyError as exc:
        raise ValueError(
            f"unknown registered checkpoint {group}/{name}/{artifact}"
        ) from exc
    return RegisteredCheckpoint(
        group=group,
        name=name,
        display_name=str(entry["display_name"]),
        artifact=artifact,
        path=project_path(artifact_entry["path"]),
        sha256=str(artifact_entry["sha256"]),
        bytes=int(artifact_entry["bytes"]),
        metadata={key: value for key, value in entry.items() if key != "artifacts"},
    )


def resolve_deepwta_checkpoint(
    profile: str = "standard",
    artifact: str = "best",
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> RegisteredCheckpoint:
    try:
        canonical = DEEPWTA_PROFILE_ALIASES[profile.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown DeepWTA profile {profile!r}") from exc
    return _resolve("deepwta", canonical, artifact, registry_path=registry_path)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_registered_checkpoint(checkpoint: RegisteredCheckpoint, *, verify_hash: bool = True) -> None:
    if not checkpoint.path.is_file():
        raise FileNotFoundError(f"registered checkpoint not found: {checkpoint.path}")
    actual_bytes = checkpoint.path.stat().st_size
    if actual_bytes != checkpoint.bytes:
        raise RuntimeError(
            f"checkpoint size mismatch for {checkpoint.display_name}/{checkpoint.artifact}: "
            f"expected {checkpoint.bytes}, got {actual_bytes}"
        )
    if verify_hash:
        actual_hash = file_sha256(checkpoint.path)
        if actual_hash != checkpoint.sha256:
            raise RuntimeError(
                f"checkpoint hash mismatch for {checkpoint.display_name}/{checkpoint.artifact}"
            )


def all_registered_checkpoints(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[RegisteredCheckpoint]:
    registry = load_checkpoint_registry(registry_path)
    records = []
    for name, entry in registry["deepwta"]["profiles"].items():
        for artifact in entry["artifacts"]:
            records.append(_resolve("deepwta", name, artifact, registry_path=registry_path))
    for group in ("auxiliary",):
        for name, entry in registry[group].items():
            for artifact in entry["artifacts"]:
                records.append(_resolve(group, name, artifact, registry_path=registry_path))
    return records
