# SDK-First Benchmark Summary

- Run id: `20260425-163450`
- Datasets: separable_topics, rare_class_trap, grouped_duplicates
- Strategies: random, entropy, group_diverse_entropy, class_balanced_entropy, class_group_balanced_entropy, margin, least_confidence, coreset_kcenter, embedding_kmeans_pp, max_min_embedding, deduplicate_near_neighbors, density_weighted_diversity, badge, mc_dropout_entropy, bald, variation_ratio, prediction_variance, committee_vote_entropy, committee_kl_divergence, committee_pairwise_disagreement, committee_margin, mix_entropy_random, mix_uncertainty_random, mix_group_diverse_random, mix_class_group_random, mix_class_group_margin_random, mix_interleaved_class_group_random, mix_interleaved_class_group_margin_random, hybrid_weighted_entropy_coreset, hybrid_uncertainty_prefilter_coreset, hybrid_diversity_prefilter_entropy, hybrid_weighted_guarded
- Budgets: 16, 32, 48, 64, 96
- Seeds: 13, 21, 34

## Best Macro-F1 By Dataset

| Dataset | Strategy | Seed | Budget | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| grouped_duplicates | badge | 13 | 32 | 1.0000 | 1.0000 |
| rare_class_trap | badge | 13 | 32 | 1.0000 | 1.0000 |
| separable_topics | badge | 13 | 32 | 1.0000 | 1.0000 |

## Stop Policy Diagnostics

| Policy | Metric | Curves | Stops | Mean Label Savings | Mean Quality Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| accuracy_plateau_conservative | accuracy | 288 | 274 | 0.3021 | -0.0009 |
| macro_f1_plateau_conservative | macro_f1 | 288 | 274 | 0.3021 | -0.0019 |
| macro_f1_plateau_fast | macro_f1 | 288 | 287 | 0.5133 | -0.0084 |

Artifacts in this directory:

- `metrics.csv`: budgeted quality metrics, AULC, lift versus random, runtime, and budget efficiency columns.
- `selections.csv`: selected ids, scheduler snapshots, label mix, duplicate counts, and group concentration diagnostics.
- `stop_policies.csv`: post-hoc stop policy decisions with label savings, quality deltas, and runtime savings.
- `manifest.json`: run configuration and SDK gap notes.
- `summary.json`: machine-readable rollup.
- `validation.json`: acquisition-surface checks for opaque ids, groups, schema, and metadata.
