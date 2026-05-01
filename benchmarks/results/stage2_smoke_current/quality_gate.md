# Quality Gate Report

- Result: PASS
- Input: `benchmarks\results\stage2_smoke_current`
- Rows: 30
- Datasets: rare_class_trap, separable_topics
- Strategies: entropy, least_confidence, margin, mix_entropy_random, random
- Budgets: 12, 24, 36
- Evidence category: `sdk_native_synthetic_diagnostic`
- Manifest: present

## Gates

| Gate | Status | Detail |
| --- | --- | --- |
| random_baseline_present | PASS | Random baseline rows are required for strategy comparisons. |
| at_least_one_non_random_strategy | PASS | Quality gates need at least one strategy to compare against random. |
| random_baseline_complete | PASS | Every non-random dataset/seed/budget row must have a matching random baseline. |
| single_strategy_quality_candidate | PASS | At least one strategy must simultaneously have non-negative final macro-F1 lift, non-negative AULC lift, non-loss-rate >= 0.75, and at least one positive final macro-F1 or AULC lift versus random. |
| non_random_non_negative_quality_lift | PASS | At least one non-random strategy must have a computable non-negative final macro-F1 or AULC lift versus matched random. |
| win_rate_available | PASS | At least one strategy-vs-random macro-F1 win-rate must be computable. |
| runtime_summary_present | PASS | Runtime summary rows with finite mean runtime must be present. |
| manifest_evidence_category_present | PASS | When manifest.json exists, the report must expose its evidence category. |
| strategy_claim_categories_separated | PASS | SDK-native, manual formula-reference, external formula-shim, and native external strategy claims must be categorized without overlap. |

## Evidence And Claim Boundaries

SDK-native, manual formula-reference, external formula-shim, and native external-library evidence are categorized separately so formula and native runtime claims are not conflated.

| Category | Strategies |
| --- | --- |
| sdk_native | `entropy`, `least_confidence`, `margin`, `mix_entropy_random`, `random` |
| manual_formula_reference | n/a |
| external_formula_shim | n/a |
| native_external | n/a |
| unknown | n/a |

## Final Budget Lift vs Random

| Dataset | Strategy | Mean Macro-F1 Lift | Std | N |
| --- | --- | ---: | ---: | ---: |
| rare_class_trap | entropy | 0.3805 | 0.0000 | 1 |
| rare_class_trap | least_confidence | 0.3805 | 0.0000 | 1 |
| rare_class_trap | margin | 0.3805 | 0.0000 | 1 |
| rare_class_trap | mix_entropy_random | 0.3673 | 0.0000 | 1 |
| separable_topics | entropy | -0.0278 | 0.0000 | 1 |
| separable_topics | least_confidence | 0.0000 | 0.0000 | 1 |
| separable_topics | margin | 0.0000 | 0.0000 | 1 |
| separable_topics | mix_entropy_random | 0.0000 | 0.0000 | 1 |

## Win Rate vs Random

| Strategy | Win Rate | Non-Loss Rate | Comparisons | Missing Baselines |
| --- | ---: | ---: | ---: | ---: |
| entropy | 0.6667 | 0.6667 | 6 | 0 |
| least_confidence | 0.6667 | 0.8333 | 6 | 0 |
| margin | 0.6667 | 0.8333 | 6 | 0 |
| mix_entropy_random | 0.6667 | 1.0000 | 6 | 0 |

## Mean AULC Lift vs Random

| Strategy | Mean AULC Lift | Std | Comparisons | Missing Baselines |
| --- | ---: | ---: | ---: | ---: |
| entropy | 0.1783 | 0.2785 | 2 | 0 |
| least_confidence | 0.1675 | 0.2977 | 2 | 0 |
| margin | 0.1715 | 0.2907 | 2 | 0 |
| mix_entropy_random | 0.1871 | 0.2614 | 2 | 0 |

## Runtime Mean

| Dataset | Strategy | Runtime Mean Seconds | Std | N |
| --- | --- | ---: | ---: | ---: |
| rare_class_trap | entropy | 0.0261 | 0.0084 | 3 |
| rare_class_trap | least_confidence | 0.0265 | 0.0074 | 3 |
| rare_class_trap | margin | 0.0263 | 0.0061 | 3 |
| rare_class_trap | mix_entropy_random | 0.0271 | 0.0120 | 3 |
| rare_class_trap | random | 0.0215 | 0.0076 | 3 |
| separable_topics | entropy | 0.0223 | 0.0041 | 3 |
| separable_topics | least_confidence | 0.0252 | 0.0089 | 3 |
| separable_topics | margin | 0.0230 | 0.0062 | 3 |
| separable_topics | mix_entropy_random | 0.0219 | 0.0077 | 3 |
| separable_topics | random | 0.0175 | 0.0058 | 3 |
