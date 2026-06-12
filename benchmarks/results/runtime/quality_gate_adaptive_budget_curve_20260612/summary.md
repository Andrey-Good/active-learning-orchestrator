# SDK-First Benchmark Summary

- Run id: `20260612-204850`
- Datasets: banking77, dair_ai_emotion
- Strategies: random, adaptive_uncertainty_diversity, entropy, class_group_balanced_entropy, badge
- Budgets: 50, 100, 200, 300, 400
- Seeds: 13, 21, 34

## Best Macro-F1 By Dataset

| Dataset | Strategy | Seed | Budget | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| banking77 | entropy | 21 | 400 | 0.3508 | 0.3920 |
| dair_ai_emotion | class_group_balanced_entropy | 34 | 300 | 0.2567 | 0.4600 |

## Stop Policy Diagnostics

| Policy | Metric | Curves | Stops | Mean Label Savings | Mean Quality Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| accuracy_plateau_conservative | accuracy | 30 | 2 | 0.0250 | -0.0024 |
| macro_f1_plateau_conservative | macro_f1 | 30 | 4 | 0.0333 | -0.0025 |
| macro_f1_plateau_fast | macro_f1 | 30 | 12 | 0.1833 | -0.0171 |

Artifacts in this directory:

- `metrics.csv`: budgeted quality metrics, AULC, lift versus random, runtime, and budget efficiency columns.
- `selections.csv`: selected ids, scheduler snapshots, label mix, duplicate counts, and group concentration diagnostics.
- `stop_policies.csv`: post-hoc stop policy decisions with label savings, quality deltas, and runtime savings.
- `full_train_reference.csv`: no-budget reference metrics from fitting on the full train split.
- `budget_warnings.csv`: requested budgets that were not executable, with explicit skip reasons.
- `manifest.json`: run configuration and SDK gap notes.
- `summary.json`: machine-readable rollup.
- `validation.json`: acquisition-surface checks for opaque ids, groups, schema, and metadata.
