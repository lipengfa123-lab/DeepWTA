# DeepWTA-Standard reproduction guide

Run all commands from the repository root.

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate deepwta
python -m pip install -e .
```

Use a PyTorch build compatible with the local CUDA driver. Data and result
checks do not require CUDA.

## 2. Validate the retained TEST-800 archive

```bash
python scripts/validate_dataset.py
python scripts/validate_release.py
```

The first command checks all 800 exported input cases. The second checks every
file listed by the Standard-only release manifest. TEST-800 retains complete
per-case metric rows but no decision NPZs because the copied files were empty
placeholders. This is represented explicitly as an expected decision count of 0.

VAL-200 retains 200 decisions and can be checked with:

```bash
python scripts/validate_release.py \
  --release-root results/reviewer2_release/val200_frozen_candidate \
  --expected-inputs 200 \
  --expected-solutions 200
```

## 3. Validate one retained Standard decision

```bash
python scripts/validate_solutions.py \
  --data results/reviewer2_release/val200_frozen_candidate/inputs/cases/idx_000000.npz \
  --result results/reviewer2_release/val200_frozen_candidate/main/deepwta/solutions/DeepWTA-Standard/idx_000000.npz
```

The validator recomputes hard feasibility and canonical metrics from the input
and decision arrays.

## 4. Reproduce Standard-only summaries

```bash
python scripts/reproduce_main_table.py
python scripts/reproduce_statistics.py
python scripts/reproduce_figures.py
```

`reproduce_main_table.py` and `reproduce_statistics.py` validate that the input
contains only `DeepWTA-Standard`, then write a one-row summary, bootstrap
confidence intervals, group summaries, and a result inventory. Cross-method
paired tests are intentionally absent.

## 5. Rerun inference

Inspect the delegated command:

```bash
python scripts/evaluate_test800.py --limit 1 --device cuda --dry-run
```

Run one case before requesting the full split:

```bash
python scripts/evaluate_test800.py --limit 1 --device cuda
python scripts/evaluate_test800.py --limit 0 --device cuda
```

New outputs are written under `reproduced_results/test800/deepwta/`.

## 6. Retained artifacts

- 800 non-empty TEST-800 input NPZ files;
- 800 TEST-800 Standard metric rows;
- 200 non-empty VAL-200 input NPZ files;
- 200 VAL-200 Standard metric rows and decision NPZ files;
- Standard checkpoints, source, configs, tests, and training data;
- SHA-256 manifests that cover retained release files.

