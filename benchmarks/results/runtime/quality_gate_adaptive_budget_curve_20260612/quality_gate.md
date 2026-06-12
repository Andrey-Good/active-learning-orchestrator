# Quality Gate Report

- Result: PASS
- Input: `benchmarks\results\runtime\quality_gate_adaptive_budget_curve_20260612`
- Rows: 150
- Datasets: banking77, dair_ai_emotion
- Strategies: adaptive_uncertainty_diversity, badge, class_group_balanced_entropy, entropy, random
- Budgets: 50, 100, 200, 300, 400
- Seed count: 3
- Evidence category: `sdk_native_capped_real_dataset`
- Real evidence level: `standard`
- Manifest: present

## Gates

| Gate | Status | Detail |
| --- | --- | --- |
| random_baseline_present | PASS | Random baseline rows are required for strategy comparisons. |
| at_least_one_non_random_strategy | PASS | Quality gates need at least one strategy to compare against random. |
| random_baseline_complete | PASS | Every non-random dataset/seed/budget row must have a matching random baseline. |
| single_strategy_quality_candidate | PASS | At least one strategy must simultaneously have non-negative final macro-F1 lift, non-negative AULC lift, meaningful-acquisition non-loss-rate >= 2/3, final-budget non-loss-rate >= 2/3, and at least one positive final macro-F1 or AULC lift versus random. Meaningful acquisition rows require selected_count >= 2 when selected_count is present. |
| non_random_non_negative_quality_lift | PASS | At least one non-random strategy must have a computable non-negative final macro-F1 or AULC lift versus matched random. |
| win_rate_available | PASS | At least one strategy-vs-random macro-F1 win-rate must be computable. |
| non_random_strategy_differentiation | PASS | When selections.csv is available, non-random strategies must not all share the same selected order in every comparable dataset/seed/budget group, and no strategy pair may be identical across all comparable groups. |
| runtime_summary_present | PASS | Runtime summary rows with finite mean runtime must be present. |
| manifest_evidence_category_present | PASS | When manifest.json exists, the report must expose its evidence category. |
| strategy_claim_categories_separated | PASS | SDK-native, manual formula-reference, external formula-shim, and native external strategy claims must be categorized without overlap. |
| stage11_real_standard_seed_count | PASS | Stage 11 standard real evidence requires at least three distinct seeds; real_smoke is smoke-only and not standard evidence. |
| stage11_real_standard_metrics_present | PASS | Stage 11 standard real metrics.csv rows must include finite calibration, coverage, and zero-recall metrics. |
| stage11_real_standard_full_train_calibration_present | PASS | Stage 11 standard real full_train_reference.csv rows must include finite calibration metrics. |

## Selection Differentiation

| Comparable groups | Collapsed selected-order groups | Collapsed effective-strategy groups | Pairwise collapsed pairs |
| ---: | ---: | ---: | ---: |
| 30 | 0 | 0 | 0 |

## Evidence And Claim Boundaries

SDK-native, manual formula-reference, external formula-shim, and native external-library evidence are categorized separately so formula and native runtime claims are not conflated.

| Category | Strategies |
| --- | --- |
| sdk_native | `adaptive_uncertainty_diversity`, `badge`, `class_group_balanced_entropy`, `entropy`, `random` |
| manual_formula_reference | n/a |
| external_formula_shim | n/a |
| native_external | n/a |
| unknown | n/a |

## Calibration At Final Budget

| Dataset | Strategy | Brier Mean | NLL Mean | ECE Mean | N |
| --- | --- | ---: | ---: | ---: | ---: |
| banking77 | adaptive_uncertainty_diversity | 0.9367 | 3.9576 | 0.3357 | 3 |
| banking77 | badge | 0.9375 | 3.8665 | 0.2998 | 3 |
| banking77 | class_group_balanced_entropy | 0.9310 | 3.6605 | 0.2965 | 3 |
| banking77 | entropy | 0.9378 | 3.7737 | 0.3097 | 3 |
| banking77 | random | 0.9342 | 3.9256 | 0.2853 | 3 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 0.7251 | 1.4965 | 0.0702 | 3 |
| dair_ai_emotion | badge | 0.7202 | 1.4870 | 0.0688 | 3 |
| dair_ai_emotion | class_group_balanced_entropy | 0.7257 | 1.4979 | 0.0686 | 3 |
| dair_ai_emotion | entropy | 0.7220 | 1.4909 | 0.0900 | 3 |
| dair_ai_emotion | random | 0.7210 | 1.4850 | 0.0750 | 3 |

## Final Budget Lift vs Random

| Dataset | Strategy | Mean Macro-F1 Lift | Std | N |
| --- | --- | ---: | ---: | ---: |
| banking77 | adaptive_uncertainty_diversity | 0.0379 | 0.0173 | 3 |
| banking77 | badge | 0.0152 | 0.0162 | 3 |
| banking77 | class_group_balanced_entropy | 0.0091 | 0.0138 | 3 |
| banking77 | entropy | 0.0233 | 0.0147 | 3 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 0.0017 | 0.0122 | 3 |
| dair_ai_emotion | badge | 0.0181 | 0.0151 | 3 |
| dair_ai_emotion | class_group_balanced_entropy | 0.0113 | 0.0144 | 3 |
| dair_ai_emotion | entropy | 0.0068 | 0.0088 | 3 |

## Win Rate vs Random

| Strategy | Win Rate | Non-Loss Rate | Comparisons | Missing Baselines |
| --- | ---: | ---: | ---: | ---: |
| adaptive_uncertainty_diversity | 0.6667 | 0.6667 | 30 | 0 |
| badge | 0.6333 | 0.6333 | 30 | 0 |
| class_group_balanced_entropy | 0.7000 | 0.7000 | 30 | 0 |
| entropy | 0.7333 | 0.7333 | 30 | 0 |

## Meaningful Acquisition Win Rate vs Random

| Strategy | Win Rate | Non-Loss Rate | Comparisons | Min Selected Count | Missing Baselines |
| --- | ---: | ---: | ---: | ---: | ---: |
| adaptive_uncertainty_diversity | 0.6667 | 0.6667 | 30 | 2 | 0 |
| badge | 0.6333 | 0.6333 | 30 | 2 | 0 |
| class_group_balanced_entropy | 0.7000 | 0.7000 | 30 | 2 | 0 |
| entropy | 0.7333 | 0.7333 | 30 | 2 | 0 |

## Final Budget Win Rate vs Random

| Strategy | Win Rate | Non-Loss Rate | Comparisons | Missing Baselines |
| --- | ---: | ---: | ---: | ---: |
| adaptive_uncertainty_diversity | 0.8333 | 0.8333 | 6 | 0 |
| badge | 0.8333 | 0.8333 | 6 | 0 |
| class_group_balanced_entropy | 0.6667 | 0.6667 | 6 | 0 |
| entropy | 1.0000 | 1.0000 | 6 | 0 |

## Mean AULC Lift vs Random

| Strategy | Mean AULC Lift | Std | Comparisons | Missing Baselines |
| --- | ---: | ---: | ---: | ---: |
| adaptive_uncertainty_diversity | 0.0196 | 0.0250 | 6 | 0 |
| badge | 0.0127 | 0.0171 | 6 | 0 |
| class_group_balanced_entropy | 0.0176 | 0.0215 | 6 | 0 |
| entropy | 0.0191 | 0.0314 | 6 | 0 |

## Runtime Mean

| Dataset | Strategy | Runtime Mean Seconds | Std | N |
| --- | --- | ---: | ---: | ---: |
| banking77 | adaptive_uncertainty_diversity | 2.4259 | 1.4096 | 15 |
| banking77 | badge | 3.7764 | 1.4023 | 15 |
| banking77 | class_group_balanced_entropy | 0.3718 | 0.2233 | 15 |
| banking77 | entropy | 1.9353 | 1.4899 | 15 |
| banking77 | random | 0.3388 | 0.1985 | 15 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 0.3037 | 0.1309 | 15 |
| dair_ai_emotion | badge | 1.0099 | 0.2217 | 15 |
| dair_ai_emotion | class_group_balanced_entropy | 0.2368 | 0.1103 | 15 |
| dair_ai_emotion | entropy | 0.3314 | 0.2302 | 15 |
| dair_ai_emotion | random | 0.2714 | 0.1218 | 15 |
