# DeepWTA Dataset

Paper: *DeepWTA: A Fast Deep Learning-Based Weapon–Target Assignment Method With Hard-Constraint Guarantee*

Complete data path: `https://drive.google.com/drive/folders/1tg3QSGjW2lbgVf59S_pk-nuHjza9bf12?usp=sharing`

## Contents

| Directory  | Cases             | Paper   |
| ---------- | ----------------- | ------- |
| `test800/` | 800               | Table 2 |
| `val200/`  | 200               | Table 3 |
| `scaling/` | LARGE 3, XLARGE 3 | Table 4 |

Each case provides: input, decision variables, objective values.

## Layout

- Input: `test800/inputs/cases/idx_XXXXXX.npz`, `val200/inputs/cases/idx_XXXXXX.npz`
- Decision variables: `outputs/<Method>/solutions/<Method>/idx_XXXXXX.npz`
- Objective values: `outputs/<Method>/test_details.csv` (val200 uses `val_details.csv`, scaling uses `case_result.csv`)

## Fields

Input NPZ: `B_inventory`, `D_distance`, `Q_intercept_prob`, `X_reachable`, `U_scheme_demand`, `G_damage_level`, `G_th_damage_threshold`, `base_coords`, `target_coords`.

Decision NPZ: `allocation[K,N,M]`, `selected_scheme[M]`.

Objective CSV: `profile_score` = J, `WR` = ammo / lower bound, `TCR` = trajectory crossing rate, `avg_distance` = Timeliness, `avg_interception` = Inter. ratio.

## Scale

K = 10 weapon types, L = 10 plans; 8 ≤ N, M ≤ 80; LARGE = 100×100, XLARGE = 200×200. `scaling/` has no input NPZ; its inputs are regenerated from `data/splits/wta_large.json` and `wta_xlarge.json`.

## Aggregated result tables

Per-method aggregates and statistics:

| Path                                                       | Content                                                                                 | Paper   |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------- |
| `test800/analysis/table_main.csv`                          | 6 methods × 800 cases: feasibility, J, WR, TCR, timeliness, interception, runtime, J CI | Table 2 |
| `test800/analysis/table_paired_statistics.csv`             | paired bootstrap / Wilcoxon / McNemar (Holm)                                            | §4.2    |
| `test800/analysis/scenario_units.csv`                      | scenario-level rows after seed averaging                                                | §4.2    |
| `test800/analysis/classical_three_seed_summary.csv`        | per-seed SA/GA/FRA-DC summary                                                           | §4.2    |
| `test800/analysis/three_group_summary.csv`                 | group A/B/C subset stability                                                            | §4.2    |
| `test800/analysis/paired_vs_deepwta.csv`                   | paired comparison vs DeepWTA                                                            | §4.2    |
| `test800/analysis/all_selected_test_rows.csv`              | every selected result row (9600 rows)                                                   | audit   |
| `val200/analysis/table_allocation_only.csv`                | 3 methods × 200 cases                                                                   | Table 3 |
| `val200/analysis/scheme_match_audit.csv`                   | frozen-scheme equality audit                                                            | §4.3    |
| `val200/analysis/paired_statistics.csv`                    | paired J differences                                                                    | §4.3    |
| `scaling/analysis/scaling_summary_large_xlarge.csv`        | 2 scales × 6 methods                                                                    | Table 4 |
| `scaling/analysis/scaling_scenario_units_large_xlarge.csv` | per-case rows behind Table 4                                                            | audit   |
| `scaling/analysis/timeout_audit_large_xlarge.csv`          | per-case status / wall-clock                                                            | audit   |

## Provenance & auxiliary

- `outputs/<Method>/protocol.json` — method protocol, checkpoint SHA-256, solver config.
- `test800/manifests/` — `RELEASE_SUMMARY.json`, `SHA256_MANIFEST.csv`.
- `scaling/logs/` — per-case run logs (LARGE / XLARGE).
- `CHECKPOINT_SHA256.csv` — released checkpoint hashes.
- `DATA_DICTIONARY.md` — full field definitions and units.

## Figures & reproduction scripts

Plotting and table-building code (from `DeepWTA_master`):

- `scripts/figures/` — figure script: `fig6_case_study.py`.
- `scripts/visualization/` — battlefield solution-map renderers: `plot_deepwta_solution_map.py`, `plot_saved_wta_solution_map.py`.
- `scripts/experiments/` — table builders and analysis: `build_final_test_tables.py`, `build_table4_scaling.py`, `analyze_main.py`, `analyze_allocation_comparison.py`, `export_cases.py`, plus evaluation drivers.
- `src/deepwta/visualization/` — academic plotting style (`academic.py`).

Generated figure:

- `results/figures/fig6_case_study/` — six methods on case `wta_val_000149`.

One-click reproduction:

```bash
python reproduce_all.py
```

Or run the figure script directly:

```bash
python scripts/figures/fig6_case_study.py
```

Full source and runnable environment live in the `DeepWTA_master` repository.

## License

CC BY 4.0 (see `LICENSE`).
