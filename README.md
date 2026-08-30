# DeepWTA-Standard

This repository is the standalone DeepWTA-Standard implementation for multi-scheme
weapon-target assignment (WTA). It contains the standard model, deterministic
integer solver, training and evaluation programs, frozen datasets, selected
checkpoints, tests, and retained Standard-only validation/test results.

Non-standard solver variants, comparison baselines, ablations, and their
artifacts are intentionally not part of this revision.

Complete data path:
https://drive.google.com/drive/folders/1tg3QSGjW2lbgVf59S_pk-nuHjza9bf12?usp=sharing

## Retained evaluation scope

- Training: 6,000 instances
- Validation: 800 instances; 200 frozen reviewer cases are retained
- Held-out test: 800 instances (`TEST-800`)
- Problem sizes: `8 <= N,M <= 80`
- Weapon types: `K = 10`
- Candidate strike plans: `L = 10`
- Tactical profile: `balanced`
- Published method label: `DeepWTA-Standard`

## Repository layout

```text
configs/       Standard model, solver, dataset, and checkpoint metadata
src/deepwta/  Standard model, solver, data, metrics, and validation code
scripts/      Standard training, inference, validation, analysis, and plotting
data/         Maps, frozen manifests, and processed training/evaluation datasets
checkpoints/  Standard checkpoints and its training-time conflict surrogate
results/      Standard-only TEST-800 and VAL-200 inputs, rows, and decisions
analysis/     Convenience copies of the Standard TEST-800 summary
docs/         Data contract, split, reproduction, release, and audit notes
tests/        Standard model, solver, data, metric, and integration tests
```

## Installation

The package supports Python 3.10-3.13. A CUDA build of PyTorch is recommended
for inference; result summarization and archive checks are CPU-only.

```bash
conda env create -f environment.yml
conda activate deepwta
python -m pip install -e .
```

## Validate retained data and results

```bash
python scripts/validate_dataset.py
python scripts/validate_release.py
```

Validate one retained Standard decision independently:

```bash
python scripts/validate_solutions.py \
  --data results/reviewer2_release/val200_frozen_candidate/inputs/cases/idx_000000.npz \
  --result results/reviewer2_release/val200_frozen_candidate/main/deepwta/solutions/DeepWTA-Standard/idx_000000.npz
```

The TEST-800 per-case metric CSV is complete. The copied TEST-800 decision NPZ
files were zero-byte placeholders and have therefore been removed instead of
being presented as valid decisions. VAL-200 retains 200 non-empty decisions.

## Reproduce Standard-only summaries

```bash
python scripts/reproduce_main_table.py
python scripts/reproduce_statistics.py
python scripts/reproduce_figures.py
```

These commands consume existing Standard-only rows and do not rerun inference.
See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for exact locations and commands.

## Rerun DeepWTA-Standard

Inspect the command first:

```bash
python scripts/evaluate_test800.py --limit 1 --device cuda --dry-run
```

Then run one case or the full frozen split:

```bash
python scripts/evaluate_test800.py --limit 1 --device cuda
python scripts/evaluate_test800.py --limit 0 --device cuda
```

New outputs go under `reproduced_results/test800/deepwta/` and do not overwrite
the retained release results.

## Integrity and limitations

- TEST-800 IDs and bytes are frozen by `data/splits/wta_e2e_v1.freeze.json`.
- Geometric crossing is a soft tactical objective, not a hard constraint.
- TEST-800 contains 800 complete metric rows but no valid copied decision NPZs.
- VAL-200 contains 200 complete metric rows and 200 non-empty decision NPZs.
- This directory has no Git metadata, so removals cannot be restored with Git.
- A project license and final citation metadata have not yet been supplied.
