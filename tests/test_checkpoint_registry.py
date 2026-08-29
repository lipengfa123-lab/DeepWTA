from __future__ import annotations

from deepwta.utils.checkpoint_registry import (
    all_registered_checkpoints,
    load_checkpoint_registry,
    resolve_deepwta_checkpoint,
    verify_registered_checkpoint,
)


def test_registry_exposes_only_standard_profile() -> None:
    registry = load_checkpoint_registry()
    assert registry["deepwta"]["input_contract"]["num_weapon_types"] == 10
    assert set(registry["deepwta"]["profiles"]) == {"standard"}


def test_standard_checkpoint_resolves_and_verifies() -> None:
    standard = resolve_deepwta_checkpoint("standard", "best")
    assert standard.display_name == "DeepWTA-Standard"
    verify_registered_checkpoint(standard, verify_hash=True)


def test_registry_contains_standard_and_training_surrogate_only() -> None:
    records = all_registered_checkpoints()
    assert len(records) == 6
    assert {record.group for record in records} == {"deepwta", "auxiliary"}
    for checkpoint in records:
        verify_registered_checkpoint(checkpoint, verify_hash=False)
