# SDK-First Benchmark Summary

- Run id: `20260426-152619`
- Datasets: dair_ai_emotion
- Strategies: random, entropy, adaptive_uncertainty_diversity, badge, hybrid_weighted_guarded
- Budgets: 50, 100, 200
- Seeds: 13, 21, 34

## Best Macro-F1 By Dataset

| Dataset | Strategy | Seed | Budget | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| dair_ai_emotion | adaptive_uncertainty_diversity | 34 | 200 | 0.2120 | 0.4433 |

## Stop Policy Diagnostics

| Policy | Metric | Curves | Stops | Mean Label Savings | Mean Quality Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| accuracy_plateau_conservative | accuracy | 15 | 1 | 0.0000 | 0.0000 |
| macro_f1_plateau_conservative | macro_f1 | 15 | 1 | 0.0000 | 0.0000 |
| macro_f1_plateau_fast | macro_f1 | 15 | 7 | 0.1667 | -0.0139 |

Artifacts in this directory:

- `metrics.csv`: budgeted quality metrics, AULC, lift versus random, runtime, and budget efficiency columns.
- `selections.csv`: selected ids, scheduler snapshots, label mix, duplicate counts, and group concentration diagnostics.
- `stop_policies.csv`: post-hoc stop policy decisions with label savings, quality deltas, and runtime savings.
- `full_train_reference.csv`: no-budget reference metrics from fitting on the full train split.
- `manifest.json`: run configuration and SDK gap notes.
- `summary.json`: machine-readable rollup.
- `validation.json`: acquisition-surface checks for opaque ids, groups, schema, and metadata.
