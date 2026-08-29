# Frozen data splits

The paper uses the `wta_e2e_v1` dataset with the following immutable split counts:

| Split | Count | Use |
|---|---:|---|
| Train | 6,000 | Model fitting only |
| Validation | 800 | Model/configuration selection and diagnostics |
| TEST | 800 | Final held-out headline evaluation |

The data-generation seed is 3026. `data/splits/wta_e2e_v1.freeze.json` records the SHA-256 and byte size of every processed shard. The manifest fixes `K=10`, `L=10`, and `8 <= N,M <= 80`.

The final release exports TEST-800 case inputs separately under `results/reviewer2_release/test800_final/inputs/cases/`. `case_index.csv` maps the unchanged integer index to `scenario_id`. Do not rename cases or renumber scenario IDs.

## VAL-200

The directory `results/reviewer2_release/val200_frozen_candidate/` contains a frozen 200-case validation subset used for Standard configuration selection and validation. It is not part of TEST-800 and must not be pooled with final test results.

## Leakage controls

- TEST-800 is not used for training or hyperparameter selection.
- The split manifest and exported cases are read-only release artifacts.
- Results must join by `scenario_index`/`scenario_id`, never by filesystem enumeration order alone.
- Stochastic algorithm seeds are aggregated within a scenario before scenario-level paired inference.
