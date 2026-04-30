# SDK-First Benchmark Summary

- Run id: `20260428-204736`
- Datasets: separable_topics, rare_class_trap
- Strategies: random, entropy, margin, least_confidence, mix_entropy_random
- Budgets: 12, 24, 36
- Seeds: 13

## Best Macro-F1 By Dataset

| Dataset | Strategy | Seed | Budget | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| rare_class_trap | entropy | 13 | 24 | 1.0000 | 1.0000 |
| separable_topics | entropy | 13 | 12 | 1.0000 | 1.0000 |

## Stop Policy Diagnostics

| Policy | Metric | Curves | Stops | Mean Label Savings | Mean Quality Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| accuracy_plateau_conservative | accuracy | 10 | 5 | 0.0000 | 0.0000 |
| macro_f1_plateau_conservative | macro_f1 | 10 | 5 | 0.0000 | 0.0000 |
| macro_f1_plateau_fast | macro_f1 | 10 | 10 | 0.1667 | -0.0163 |

Artifacts in this directory:

- `metrics.csv`: budgeted quality metrics, AULC, lift versus random, runtime, and budget efficiency columns.
- `selections.csv`: selected ids, scheduler snapshots, label mix, duplicate counts, and group concentration diagnostics.
- `stop_policies.csv`: post-hoc stop policy decisions with label savings, quality deltas, and runtime savings.
- `full_train_reference.csv`: no-budget reference metrics from fitting on the full train split.
- `manifest.json`: run configuration and SDK gap notes.
- `summary.json`: machine-readable rollup.
- `validation.json`: acquisition-surface checks for opaque ids, groups, schema, and metadata.
