# DeepWTA-Standard cleanup audit

Audit date: 2026-08-29 (Asia/Shanghai).

## Retained scope

| Category | Location |
|---|---|
| Standard model and graph encoder | `src/deepwta/models/` |
| Standard deterministic solver | `src/deepwta/solver.py` and `src/deepwta/mechanisms/` |
| Standard training | `scripts/train/train_deepwta.py` |
| Optional conflict surrogate used by training | `src/deepwta/models/trajectory_conflict_gnn.py` |
| Standard evaluation | `scripts/evaluate/evaluate_deepwta.py` and `scripts/reviewer2/evaluate_deepwta_release.py` |
| Data generation/loading | `src/deepwta/data/` and `scripts/datagen/` |
| Metrics and validation | `src/deepwta/evaluation/`, `src/deepwta/benchmark/`, and `src/deepwta/utils/validator.py` |
| Standard tests | `tests/` |
| Standard results | `results/reviewer2_release/` |

## Removed scope

- non-standard solver source, configs, scripts, tests, checkpoints, and results;
- comparison implementations and artifacts;
- non-standard public configuration, evaluation paths, and tests;
- allocation-only comparisons, ablations, comparison figures, and galleries;
- historical/local `Else/` archive and stale package metadata.

## Result integrity

- TEST-800: 800 unique Standard rows are retained. The 800 copied decision NPZ
  files were all zero bytes, so they were removed and are not claimed as valid.
- VAL-200: 200 unique Standard rows and 200 non-empty decision NPZ files are retained.
- Release manifests are regenerated after cleanup and exclude their own manifest
  and summary metadata to avoid self-referential hashes.

## Repository state

This directory has no `.git` metadata. Cleanup changes therefore cannot be
recovered with `git restore`; external backups remain the only rollback source.
