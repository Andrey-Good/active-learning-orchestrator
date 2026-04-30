# Reference Strategy Benchmark Summary

- Run id: `20260425-163618`
- Datasets: separable_topics, rare_class_trap, grouped_duplicates
- Strategies: random, entropy, margin, least_confidence, class_group_balanced_entropy, mix_interleaved_class_group_random, manual_entropy, manual_margin, manual_least_confidence, manual_class_group_balanced_entropy, manual_random, modal_entropy, modal_margin, modal_uncertainty, skactiveml_entropy, skactiveml_margin, skactiveml_least_confidence
- Budgets: 16, 32, 48, 64, 96
- Seeds: 13, 21, 34
- Mean SDK/manual formula-equivalence Jaccard: 0.9855374466313579

## Best Macro-F1 By Dataset

| Dataset | Strategy | Family | Seed | Budget | Macro-F1 | Accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| grouped_duplicates | class_group_balanced_entropy | sdk | 13 | 32 | 1.0000 | 1.0000 |
| rare_class_trap | class_group_balanced_entropy | sdk | 13 | 16 | 1.0000 | 1.0000 |
| separable_topics | class_group_balanced_entropy | sdk | 13 | 32 | 1.0000 | 1.0000 |

Artifacts in this directory:

- `metrics.csv`: learning-curve quality metrics, AULC, early macro-F1, rare recall, and runtime.
- `selections.csv`: selected ids and group concentration diagnostics.
- `equivalence.csv`: formula-equivalent SDK/manual overlap, Jaccard, and exact order diagnostics.
- `external_adapters.json`: optional external-library availability and skip reasons.
- `manifest.json`: run configuration and benchmark contract.
