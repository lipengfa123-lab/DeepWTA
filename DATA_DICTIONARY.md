# Data Dictionary

Notation: `K = 10` weapon types, `L = 10` candidate strike plans, `N` bases, `M` targets. `N` and `M` vary per case (`8 <= N, M <= 80`).
Indices are zero-based. Distances use the map coordinate units of `base_coords` / `target_coords`; times are milliseconds; ratios are dimensionless.

---

## 1. Input case file — `inputs/cases/idx_XXXXXX.npz`

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `base_coords` | `(N, 2)` | float64 | base coordinates `(x, y)` |
| `target_coords` | `(M, 2)` | float64 | target coordinates `(x, y)` |
| `D_distance` | `(N, M)` | float64 | base-target distance |
| `Q_intercept_prob` | `(K, N, M)` | float64 | interception probability of weapon `k` from base `i` against target `j` |
| `X_reachable` | `(K, N, M)` | float64 | reachability mask (1 = engageable) |
| `B_inventory` | `(K, N)` | float64 | inventory of weapon `k` at base `i` |
| `U_scheme_demand` | `(L, K, M)` | float64 | weapons of type `k` required by target `j` under strike plan `l` |
| `G_damage_level` | `(L, M)` | float64 | damage level of plan `l` on target `j` |
| `G_th_damage_threshold` | `(M,)` | float64 | required damage threshold of target `j` |
| `scheme_resource_AUDIT_ONLY` | `(L, M)` | float64 | total weapon count required by plan `l` for target `j` |
| `reference_scheme_AUDIT_ONLY` | `(M,)` | int64 | reference feasible strike plan (audit witness) |
| `reference_allocation_AUDIT_ONLY` | `(K, N, M)` | int64 | reference feasible allocation (audit witness) |
| `scenario_index` | scalar | int64 | global case index |
| `scenario_id` | scalar | str | case identifier, e.g. `wta_test_000143` |
| `scenario_seed` | scalar | int64 | generation seed of the case |
| `group_id` | scalar | str | `A`/`B`/`C` TEST-800 subset label |
| `split` | scalar | str | `train` / `val` / `test` |
| `dataset_name` | scalar | str | `wta_e2e_v1` |

`reference_*_AUDIT_ONLY` arrays are **not** model outputs. They are witnesses used to certify that each generated case admits a feasible solution.

`inputs/case_index.csv` repeats the identity fields and adds `K_weapon_types`, `N_bases`, `M_targets`, `L_schemes`, `input_npz` and the SHA-256 of every input file. `inputs/dataset_meta.json` records the split size and the generation configuration.

---

## 2. Solution file — `outputs/<Method>/solutions/.../idx_XXXXXX.npz`

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `selected_scheme` | `(M,)` | int64 | chosen strike plan `l` for each target |
| `allocation` | `(K, N, M)` | int64 | number of weapons of type `k` sent from base `i` to target `j` |
| `metadata_json` | scalar | str | JSON string with the method/protocol metadata |
| `scenario_index` | scalar | int64 | case index |
| `scenario_id` | scalar | str | case identifier |
| `group_id` | scalar | str | TEST-800 subset label |

These two arrays are the complete decision variables of the paper: the
strike-plan selection `c` and the integer weapon allocation `A`.

---

## 3. Evaluation protocol — `outputs/<Method>/protocol.json`

| Key | Meaning |
|---|---|
| `protocol` | protocol identifier |
| `method` | evaluated method |
| `checkpoint` | checkpoint path recorded at evaluation time |
| `checkpoint_sha256` | SHA-256 of the checkpoint |
| `dataset_manifest` | manifest used for evaluation (DeepWTA) |
| `split` | `test` or `val` |
| `count` | number of evaluated cases |
| `solver_config` | full DeepWTA calibration/solver settings (DeepWTA only) |
| `decision_variables` | path pattern of the released decision variables |
| `inference_only`, `optimizer_loaded` | inference protocol flags (learning baselines) |

---

## 4. Per-case results — `test_details.csv` / `val_details.csv`

Each row is one evaluated `(method, scenario, seed)` triple.

**Identity**

| Column | Meaning |
|---|---|
| `scenario_index`, `scenario_id` | case identity |
| `group_id`, `scenario_seed` | subset label and generation seed |
| `N`, `M` | number of bases / targets |
| `method`, `method_seed` | method name and run seed (search-based methods) |

**Hard feasibility**

| Column | Meaning |
|---|---|
| `hard_feasible` | 1 only if all hard constraints hold |
| `inventory_feasible` | no inventory overflow |
| `demand_feasible` | weapon demand satisfied |
| `level_feasible` | damage-level requirement satisfied |
| `reachability_feasible` | no unreachable assignment |
| `integer_feasible` | allocation is integral and non-negative |

**Weapon consumption and trajectory crossings**

| Column | Meaning |
|---|---|
| `ammo` | total weapons used |
| `ammo_norm` | weapons used divided by total inventory |
| `raw_ammo_capacity_lb` | raw lower bound on required weapons |
| `WR` | **WR** (paper): `ammo / raw_ammo_capacity_lb` |
| `crossing_count` | exact number of crossing trajectory pairs |
| `eligible_active_route_pairs` | number of route pairs that could cross |
| `TCR` | **TCR** (paper): `crossing_count / eligible_active_route_pairs` |
| `crossings_per_active_route` | crossings normalized by active routes |
| `active_route_count`, `active_route_ratio` | number / fraction of active base-target routes |

**Geometry, risk and violations**

| Column | Meaning |
|---|---|
| `avg_distance` | average route length (paper: Timeliness) |
| `avg_interception` | average interception probability (paper: Inter. ratio) |
| `demand_violation`, `max_demand_shortfall` | unmet demand |
| `inventory_violation`, `max_inventory_violation` | inventory overflow |
| `reachability_violation` | count of unreachable assignments |

**Allocation structure**

`active_base_count`, `active_base_ratio`, `mean_active_base_load`,
`max_base_load_ratio`, `base_load_gini`, `mean_inventory_utilization`,
`max_inventory_utilization`, `mean_damage_level`, `level1_ratio`,
`level2_ratio`, `level3_ratio`.

**Objective (canonical tactical objective J)**

| Column | Meaning |
|---|---|
| `profile_score` | **J** |
| `profile_ammo_term`, `profile_crossing_term`, `profile_distance_term`, `profile_interception_term` | v1-v4 terms before weighting |
| `profile_ammo_contribution`, `profile_crossing_contribution`, `profile_distance_contribution`, `profile_interception_contribution` | weighted contributions; `J` is their sum |
| `profile_weight_*` | weights (0.32 / 0.28 / 0.22 / 0.18) |
| `profile_scale_*` | normalization scales |
| `legacy_*` | legacy profile decomposition kept for audit only |

**Runtime**

| Column | Meaning |
|---|---|
| `network_time_ms` | model forward pass |
| `runtime_ms` | decision time (method dependent) |
| `total_runtime_ms` | end-to-end time; `runtime_scope` records what is included |
| `construction_fallback_used` | whether the calibration fallback was used |

**Search-based diagnostics**

Columns prefixed with `diag_` (and `iterations`, `best_iteration`,
`evaluations`, `stop_reason`, `solver_seed`, `base_budget_profile`) document the
SA-MC / GA-MC / FRA-DC search. They are diagnostics, not paper metrics.

---

## 5. Aggregated analysis tables

### `test800/analysis/table_main.csv` (Table 2)

| Column | Meaning |
|---|---|
| `Method` | method name |
| `N` | number of scenarios (800) |
| `Feasible` | hard-feasibility rate |
| `J_FeasibleMean` | J averaged over feasible cases |
| `J_AllMean` | J averaged over all cases (diagnostic) |
| `WR` | weapon-consumption ratio to lower bound |
| `TCR` | trajectory-crossing rate |
| `Timeliness`, `Inter. ratio` | timeliness and interception ratio |
| `RuntimeMeanMs`, `RuntimeMedianMs`, `RuntimeP95Ms` | runtime statistics |
| `J_CI95_Low`, `J_CI95_High` | paired bootstrap CI of J |

### `test800/analysis/table_paired_statistics.csv`

`Reference`, `Baseline`, `PairedN`, `BothFeasibleN`,
`DeltaJ_BaselineMinusReference`, `DeltaJ_CI95_Low/High`, `WilcoxonP_Raw`,
feasibility disagreement counts, `McNemarExactP_Raw`, and Holm-corrected
variants.

### Other files

- `classical_three_seed_summary.csv`: `method, method_seed, hard_feasible,
  profile_score, TCR, WR, runtime_ms`.
- `three_group_summary.csv`: the same metrics aggregated over groups A/B/C.
- `paired_vs_deepwta.csv`: paired comparisons against DeepWTA.
- `scenario_units.csv`: scenario-level rows after averaging the three seeds.
- `all_selected_test_rows.csv`: every selected result row (9600 rows).
- `val200_allocation_only/analysis/table_allocation_only.csv`: Table 3
  (`Method, N, Feasible, J, WR, TCR, Timeliness, Inter. ratio, RuntimeMedianMs,
  RuntimeP95Ms`).
- `val200_allocation_only/analysis/scheme_match_audit.csv`: verifies that the
  DeepWTA selected scheme equals the frozen scheme used by the allocation-only
  baselines.
- `val200_allocation_only/analysis/comparison_rows.csv`: per-case rows for the
  three allocation-only methods.

---

## 6. Scaling summary — `scaling/analysis/scaling_summary_large_xlarge.csv`

| Column | Meaning |
|---|---|
| `Scale` | `LARGE` (100x100) or `XLARGE` (200x200) |
| `Method` | method name |
| `ExpectedCases`, `CompletedCases`, `TimeoutCases`, `ErrorCases` | run accounting (3 cases per scale) |
| `CompletionRate` | completed / expected |
| `HardFeasibleConditional` | feasibility among completed cases |
| `EndToEndSuccessRate` | feasibility over all expected cases |
| `J_FeasibleMean` | J over feasible cases |
| `WR` | weapon-consumption ratio |
| `TCR` | trajectory-crossing rate |
| `Timeliness`, `Inter. ratio` | timeliness and interception ratio |
| `RuntimeMedianMs_completed_only`, `RuntimeP95Ms_completed_only` | runtime of completed cases |
| `TimeoutCapMs` | 1-hour timeout cap |

`scaling_scenario_units_large_xlarge.csv` and
`all_scaling_case_rows_large_xlarge.csv` provide the per-case audit trail.
`timeout_audit_large_xlarge.csv` records the wall-clock time and status of every
case.

---

## 7. Conventions and caveats

1. **`WR`** and **`TCR`** are used as field names everywhere in the release:
   per-case files, aggregate tables and figure labels all match the paper.
2. **Seed aggregation**: `scenario_units.csv` and all paper statistics average
   seed runs within a scenario first, so the scenario is the statistical unit.
3. **Feasibility first**: J is only meaningful for feasible cases.
   `J_FeasibleMean` is the paper's J; `J_AllMean` is diagnostic. For
   MultiHead-DQN and PointerNet-AC the paper's J is the feasible-case mean.
4. **Timeliness** (paper) is `avg_distance` in the per-case files and
   `Timeliness` in the aggregate tables; **Inter. ratio** (paper) is
   `avg_interception` in the per-case files and `Inter. ratio` in the aggregate
   tables.
5. Learning baselines are frozen-checkpoint evaluations, so their `method_seed`
   column is constant and their results have no seed variance.

---

## 8. Fig. 6 case study — `fig6_case_study/`

The manuscript's Fig. 6 uses one battlefield instance
(`wta_val_000149`, `scenario_index = 149`) with six method panels.

| Path | Content |
|---|---|
| `inputs/idx_000149.npz` | the case input, same keys as §1 |
| `solutions/<Method>/idx_000149.npz` | the decision variables, same keys as §2 |

`<Method>` is one of `DeepWTA-Standard`, `MultiHead-DQN-E2E`,
`PointerNet-AC-E2E`, `FRA-DC`, `SA-MC`, `GA-MC`. The classical methods use
seed 2026. The panels are rendered by `python scripts/figures/fig6_case_study.py`.
