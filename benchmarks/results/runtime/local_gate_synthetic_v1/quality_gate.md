# Quality Gate Report

- Result: PASS
- Input: `benchmarks\results\runtime\local_gate_synthetic_v1`
- Rows: 405
- Datasets: grouped_duplicates, rare_class_trap, separable_topics
- Strategies: badge, class_group_balanced_entropy, coreset_kcenter, density_weighted_diversity, entropy, least_confidence, margin, mix_interleaved_class_group_random, random
- Budgets: 16, 32, 48, 64, 96

## Gates

| Gate | Status | Detail |
| --- | --- | --- |
| random_baseline_present | PASS | Random baseline rows are required for strategy comparisons. |
| at_least_one_non_random_strategy | PASS | Quality gates need at least one strategy to compare against random. |
| random_baseline_complete | PASS | Every non-random dataset/seed/budget row must have a matching random baseline. |
| single_strategy_quality_candidate | PASS | At least one strategy must simultaneously have non-negative final macro-F1 lift, non-negative AULC lift, non-loss-rate >= 0.75, and at least one positive final macro-F1 or AULC lift versus random. |
| win_rate_available | PASS | At least one strategy-vs-random macro-F1 win-rate must be computable. |

## Final Budget Lift vs Random

| Dataset | Strategy | Mean Macro-F1 Lift | Std | N |
| --- | --- | ---: | ---: | ---: |
| grouped_duplicates | badge | 0.0000 | 0.0000 | 3 |
| grouped_duplicates | class_group_balanced_entropy | 0.0000 | 0.0000 | 3 |
| grouped_duplicates | coreset_kcenter | 0.0000 | 0.0000 | 3 |
| grouped_duplicates | density_weighted_diversity | 0.0000 | 0.0000 | 3 |
| grouped_duplicates | entropy | 0.0000 | 0.0000 | 3 |
| grouped_duplicates | least_confidence | 0.0000 | 0.0000 | 3 |
| grouped_duplicates | margin | 0.0000 | 0.0000 | 3 |
| grouped_duplicates | mix_interleaved_class_group_random | 0.0000 | 0.0000 | 3 |
| rare_class_trap | badge | 0.0127 | 0.0220 | 3 |
| rare_class_trap | class_group_balanced_entropy | 0.0127 | 0.0220 | 3 |
| rare_class_trap | coreset_kcenter | -0.0649 | 0.1026 | 3 |
| rare_class_trap | density_weighted_diversity | 0.0127 | 0.0220 | 3 |
| rare_class_trap | entropy | 0.0171 | 0.0296 | 3 |
| rare_class_trap | least_confidence | 0.0171 | 0.0296 | 3 |
| rare_class_trap | margin | 0.0127 | 0.0220 | 3 |
| rare_class_trap | mix_interleaved_class_group_random | 0.0127 | 0.0220 | 3 |
| separable_topics | badge | 0.0000 | 0.0000 | 3 |
| separable_topics | class_group_balanced_entropy | 0.0000 | 0.0000 | 3 |
| separable_topics | coreset_kcenter | 0.0000 | 0.0000 | 3 |
| separable_topics | density_weighted_diversity | 0.0000 | 0.0000 | 3 |
| separable_topics | entropy | 0.0000 | 0.0000 | 3 |
| separable_topics | least_confidence | 0.0000 | 0.0000 | 3 |
| separable_topics | margin | 0.0000 | 0.0000 | 3 |
| separable_topics | mix_interleaved_class_group_random | 0.0000 | 0.0000 | 3 |

## Win Rate vs Random

| Strategy | Win Rate | Non-Loss Rate | Comparisons | Missing Baselines |
| --- | ---: | ---: | ---: | ---: |
| badge | 0.3556 | 1.0000 | 45 | 0 |
| class_group_balanced_entropy | 0.3778 | 1.0000 | 45 | 0 |
| coreset_kcenter | 0.3333 | 0.8667 | 45 | 0 |
| density_weighted_diversity | 0.3111 | 0.8889 | 45 | 0 |
| entropy | 0.3333 | 0.8222 | 45 | 0 |
| least_confidence | 0.2667 | 0.8222 | 45 | 0 |
| margin | 0.2222 | 0.8000 | 45 | 0 |
| mix_interleaved_class_group_random | 0.4000 | 1.0000 | 45 | 0 |

## Mean AULC Lift vs Random

| Strategy | Mean AULC Lift | Std | Comparisons | Missing Baselines |
| --- | ---: | ---: | ---: | ---: |
| badge | 0.0418 | 0.0506 | 9 | 0 |
| class_group_balanced_entropy | 0.0447 | 0.0544 | 9 | 0 |
| coreset_kcenter | 0.0305 | 0.0291 | 9 | 0 |
| density_weighted_diversity | 0.0243 | 0.0541 | 9 | 0 |
| entropy | 0.0190 | 0.0527 | 9 | 0 |
| least_confidence | 0.0217 | 0.0583 | 9 | 0 |
| margin | -0.0043 | 0.0463 | 9 | 0 |
| mix_interleaved_class_group_random | 0.0439 | 0.0542 | 9 | 0 |

## Runtime Mean

| Dataset | Strategy | Runtime Mean Seconds | Std | N |
| --- | --- | ---: | ---: | ---: |
| grouped_duplicates | badge | 0.2111 | 0.0151 | 15 |
| grouped_duplicates | class_group_balanced_entropy | 0.0274 | 0.0022 | 15 |
| grouped_duplicates | coreset_kcenter | 0.0677 | 0.0171 | 15 |
| grouped_duplicates | density_weighted_diversity | 0.0522 | 0.0087 | 15 |
| grouped_duplicates | entropy | 0.0287 | 0.0021 | 15 |
| grouped_duplicates | least_confidence | 0.0280 | 0.0019 | 15 |
| grouped_duplicates | margin | 0.0288 | 0.0050 | 15 |
| grouped_duplicates | mix_interleaved_class_group_random | 0.0280 | 0.0026 | 15 |
| grouped_duplicates | random | 0.0248 | 0.0020 | 15 |
| rare_class_trap | badge | 0.1976 | 0.0113 | 15 |
| rare_class_trap | class_group_balanced_entropy | 0.0343 | 0.0038 | 15 |
| rare_class_trap | coreset_kcenter | 0.0694 | 0.0180 | 15 |
| rare_class_trap | density_weighted_diversity | 0.0649 | 0.0128 | 15 |
| rare_class_trap | entropy | 0.0318 | 0.0020 | 15 |
| rare_class_trap | least_confidence | 0.0323 | 0.0023 | 15 |
| rare_class_trap | margin | 0.0339 | 0.0032 | 15 |
| rare_class_trap | mix_interleaved_class_group_random | 0.0325 | 0.0027 | 15 |
| rare_class_trap | random | 0.0283 | 0.0023 | 15 |
| separable_topics | badge | 0.1958 | 0.0143 | 15 |
| separable_topics | class_group_balanced_entropy | 0.0294 | 0.0028 | 15 |
| separable_topics | coreset_kcenter | 0.0668 | 0.0184 | 15 |
| separable_topics | density_weighted_diversity | 0.0598 | 0.0108 | 15 |
| separable_topics | entropy | 0.0284 | 0.0023 | 15 |
| separable_topics | least_confidence | 0.0296 | 0.0029 | 15 |
| separable_topics | margin | 0.0304 | 0.0036 | 15 |
| separable_topics | mix_interleaved_class_group_random | 0.0301 | 0.0038 | 15 |
| separable_topics | random | 0.0254 | 0.0029 | 15 |
