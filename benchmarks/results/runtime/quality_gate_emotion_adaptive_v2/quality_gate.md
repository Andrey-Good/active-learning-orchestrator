# Quality Gate Report

- Result: PASS
- Input: `benchmarks\results\runtime\quality_gate_emotion_adaptive_v2`
- Rows: 45
- Datasets: dair_ai_emotion
- Strategies: adaptive_uncertainty_diversity, badge, entropy, hybrid_weighted_guarded, random
- Budgets: 50, 100, 200

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
| dair_ai_emotion | adaptive_uncertainty_diversity | 0.0273 | 0.0145 | 3 |
| dair_ai_emotion | badge | 0.0063 | 0.0072 | 3 |
| dair_ai_emotion | entropy | 0.0184 | 0.0176 | 3 |
| dair_ai_emotion | hybrid_weighted_guarded | -0.0053 | 0.0142 | 3 |

## Win Rate vs Random

| Strategy | Win Rate | Non-Loss Rate | Comparisons | Missing Baselines |
| --- | ---: | ---: | ---: | ---: |
| adaptive_uncertainty_diversity | 0.8889 | 0.8889 | 9 | 0 |
| badge | 0.6667 | 0.6667 | 9 | 0 |
| entropy | 0.7778 | 0.7778 | 9 | 0 |
| hybrid_weighted_guarded | 0.6667 | 0.6667 | 9 | 0 |

## Mean AULC Lift vs Random

| Strategy | Mean AULC Lift | Std | Comparisons | Missing Baselines |
| --- | ---: | ---: | ---: | ---: |
| adaptive_uncertainty_diversity | 0.0187 | 0.0139 | 3 | 0 |
| badge | 0.0196 | 0.0236 | 3 | 0 |
| entropy | 0.0229 | 0.0028 | 3 | 0 |
| hybrid_weighted_guarded | 0.0149 | 0.0150 | 3 | 0 |

## Runtime Mean

| Dataset | Strategy | Runtime Mean Seconds | Std | N |
| --- | --- | ---: | ---: | ---: |
| dair_ai_emotion | adaptive_uncertainty_diversity | 0.1062 | 0.0351 | 9 |
| dair_ai_emotion | badge | 0.3636 | 0.0255 | 9 |
| dair_ai_emotion | entropy | 0.0805 | 0.0252 | 9 |
| dair_ai_emotion | hybrid_weighted_guarded | 0.3704 | 0.2256 | 9 |
| dair_ai_emotion | random | 0.0758 | 0.0227 | 9 |
