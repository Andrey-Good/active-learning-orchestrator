# SDK-First Benchmark Summary

- Run id: `20260426-143138`
- Datasets: separable_topics, rare_class_trap, grouped_duplicates
- Strategies: random, entropy, margin, least_confidence, class_group_balanced_entropy, mix_interleaved_class_group_random, coreset_kcenter, badge, density_weighted_diversity
- Budgets: 16, 32, 48, 64, 96
- Seeds: 13, 21, 34

## Best Macro-F1 By Dataset

| Dataset | Strategy | Seed | Budget | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| grouped_duplicates | badge | 13 | 16 | 1.0000 | 1.0000 |
| rare_class_trap | badge | 13 | 32 | 1.0000 | 1.0000 |
| separable_topics | badge | 13 | 16 | 1.0000 | 1.0000 |

## Stop Policy Diagnostics

| Policy | Metric | Curves | Stops | Mean Label Savings | Mean Quality Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| accuracy_plateau_conservative | accuracy | 81 | 76 | 0.2881 | 0.0009 |
| macro_f1_plateau_conservative | macro_f1 | 81 | 76 | 0.2881 | 0.0024 |
| macro_f1_plateau_fast | macro_f1 | 81 | 81 | 0.5021 | -0.0009 |

Artifacts in this directory:

- `metrics.csv`: budgeted quality metrics, AULC, lift versus random, runtime, and budget efficiency columns.
- `selections.csv`: selected ids, scheduler snapshots, label mix, duplicate counts, and group concentration diagnostics.
- `stop_policies.csv`: post-hoc stop policy decisions with label savings, quality deltas, and runtime savings.
- `full_train_reference.csv`: no-budget reference metrics from fitting on the full train split.
- `manifest.json`: run configuration and SDK gap notes.
- `summary.json`: machine-readable rollup.
- `validation.json`: acquisition-surface checks for opaque ids, groups, schema, and metadata.
