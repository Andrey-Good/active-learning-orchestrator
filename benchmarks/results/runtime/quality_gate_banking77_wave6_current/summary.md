# SDK-First Benchmark Summary

- Run id: `20260429-231557`
- Datasets: banking77
- Strategies: random, entropy, margin, least_confidence, class_group_balanced_entropy, coreset_kcenter, adaptive_uncertainty_diversity, mix_interleaved_class_group_random
- Budgets: 50, 100, 200
- Seeds: 13, 21, 34

## Best Macro-F1 By Dataset

| Dataset | Strategy | Seed | Budget | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| banking77 | least_confidence | 21 | 200 | 0.2178 | 0.2520 |

## Stop Policy Diagnostics

| Policy | Metric | Curves | Stops | Mean Label Savings | Mean Quality Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| accuracy_plateau_conservative | accuracy | 24 | 0 | 0.0000 | 0.0000 |
| macro_f1_plateau_conservative | macro_f1 | 24 | 0 | 0.0000 | 0.0000 |
| macro_f1_plateau_fast | macro_f1 | 24 | 2 | 0.0417 | -0.0073 |

Artifacts in this directory:

- `metrics.csv`: budgeted quality metrics, AULC, lift versus random, runtime, and budget efficiency columns.
- `selections.csv`: selected ids, scheduler snapshots, label mix, duplicate counts, and group concentration diagnostics.
- `stop_policies.csv`: post-hoc stop policy decisions with label savings, quality deltas, and runtime savings.
- `full_train_reference.csv`: no-budget reference metrics from fitting on the full train split.
- `budget_warnings.csv`: requested budgets that were not executable, with explicit skip reasons.
- `manifest.json`: run configuration and SDK gap notes.
- `summary.json`: machine-readable rollup.
- `validation.json`: acquisition-surface checks for opaque ids, groups, schema, and metadata.
