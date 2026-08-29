# Data and solution format

Each exported case and solution is a compressed NumPy archive (`.npz`). Arrays use zero-based indices. For a case with `N` bases, `M` targets, `K=10` weapon types, and `L=10` strike plans:

| Public field | Shape | dtype in release | Meaning | Paper symbol / unit |
|---|---|---|---|---|
| `scenario_index` | scalar | `int64` | Stable TEST-800 row index | case index |
| `scenario_id` | scalar | Unicode | Stable scenario identifier | scenario ID |
| `base_coords` | `[N,2]` | `float64` | Base coordinates | base position; map-coordinate unit |
| `target_coords` | `[M,2]` | `float64` | Target coordinates | target position; map-coordinate unit |
| `B_inventory` | `[K,N]` | `float64` | Available inventory by weapon type and base | `B` / shots |
| `U_scheme_demand` | `[L,K,M]` | `float64` | Required effective weapon demand for each candidate plan | `U` / effective shots |
| `G_damage_level` | `[L,M]` | `float64` | Damage level delivered by each plan | `G` / ordinal level |
| `G_th_damage_threshold` | `[M]` | `float64` | Minimum required target damage level | `G_th` / ordinal level |
| `Q_intercept_prob` | `[K,N,M]` | `float64` | Interception probability for a weapon route | `Q` / probability |
| `D_distance` | `[N,M]` | `float64` | Euclidean route distance | `D` / map-distance unit |
| `X_reachable` | `[K,N,M]` | `float64` | Route legality indicator (`1` reachable) | `X` / binary |
| `selected_scheme` | `[M]` | `int64` | Chosen plan index for each target | decision variable |
| `allocation` | `[K,N,M]` | `int64` | Number of weapons assigned by type/base/target | decision variable / shots |

Input archives also contain `group_id`, `dataset_name`, `split`, and `scenario_seed`. Fields ending in `_AUDIT_ONLY` are generation witnesses used for auditing and must not be treated as solver inputs or ground-truth optimal solutions.

Solution archives may additionally contain `selected_scheme_raw`, `allocation_raw`, and `metadata_json`. The canonical final decision is always `selected_scheme` plus `allocation`.

## Hard feasibility

`src/deepwta/utils/validator.py` is the canonical validator. It checks:

1. shape, finiteness, and data domains;
2. non-negativity and integrality of `allocation`;
3. inventory limits;
4. interception-adjusted plan demand;
5. selected-plan damage threshold;
6. route reachability.

Route crossings are reported but are a soft objective, not a hard constraint.

## Metrics and canonical objective

For the frozen `balanced` tactical profile, lower is better:

```text
J = 0.32 * (Ammo / raw-ammo lower bound)
  + 0.28 * (PCR / 0.10)
  + 0.22 * (average normalized distance / 0.515)
  + 0.18 * (average interception probability / 0.132)
```

`PCR` is the exact crossing count divided by the number of eligible unordered active-route pairs (pairs sharing a base or target are excluded). Distance is normalized within the case by the maximum value in `D_distance`. The ammo denominator is the method-independent optimistic capacity lower bound implemented in `src/deepwta/evaluation/profile_objective.py`.

Per-case CSV files also include feasibility flags, raw metric components, runtime scope, algorithm seed, and method-specific diagnostics. Column names are preserved exactly; `scenario_units.csv` is the scenario-level aggregation used for paired inference.
