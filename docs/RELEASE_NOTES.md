# Release notes

## DeepWTA-Standard-only revision

- Retains only the DeepWTA-Standard product surface and its direct training,
  evaluation, data, checkpoint, validation, visualization, and test assets.
- Retains the conflict-GNN surrogate because it is an optional dependency of
  Standard training.
- Removes non-standard variants, comparison baselines, ablations, historical
  archives, and their checkpoints/results.
- Replaces multi-method paired analysis with Standard-only descriptive tables,
  bootstrap confidence intervals, group summaries, and metric distributions.
- Removes zero-byte TEST-800 decision placeholders; the complete 800-row metric
  result is retained. VAL-200 retains 200 valid decisions.
- Rebuilds release SHA-256 manifests for the retained contents.

No model checkpoint, Standard per-case metric row, or valid Standard decision
was numerically edited during the cleanup.
