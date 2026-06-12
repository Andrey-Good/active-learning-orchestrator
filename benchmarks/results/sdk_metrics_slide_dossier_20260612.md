# Active Learning SDK Metrics Slide Dossier

Generated: 2026-06-12

This single file is intended as the technical source for slides about the SDK metric results. It preserves the benchmark setup, headline statistics, caveats, chart-ready data map, and slide-story options so a deck can be assembled without re-reading every CSV artifact.

No raster charts, screenshots, or fixed visual designs are embedded here. All visual recommendations are text-only chart contracts so the slide design can define its own style.

## Technical Summary

- Banking77: adaptive at budget 400 reached 0.3199 macro-F1, matching capped full-train reference 0.3191 and beating random by 0.0379.
- Banking77: at budget 200 adaptive macro-F1 was 0.1962, a +0.0493 lift over random before the full budget was reached.
- DAIR AI Emotion: adaptive ended at 0.1744 macro-F1, within 0.0072 of full train; badge was the strongest final active strategy at 0.1908, +0.0181 over random.
- Evidence boundary: these metrics validate SDK acquisition strategy behavior using the benchmark-owned scikit-learn TF-IDF/logistic-regression adapter. They do not by themselves prove neural/Transformer production parity or CUDA acceleration.

## Source Inventory

| Purpose | Path |
| --- | --- |
| Promoted benchmark metrics | `benchmarks/results/runtime/quality_gate_adaptive_budget_curve_20260612/metrics.csv` |
| Promoted full-train reference | `benchmarks/results/runtime/quality_gate_adaptive_budget_curve_20260612/full_train_reference.csv` |
| Promoted quality gate | `benchmarks/results/runtime/quality_gate_adaptive_budget_curve_20260612/quality_gate.md` and `quality_gate.json` |

## Benchmark Contract

| Field | Value |
| --- | --- |
| Preset | real_medium |
| Evidence category | active_learning_quality |
| Real evidence level | standard |
| Datasets used for main slide evidence | banking77, dair_ai_emotion |
| Strategies in promoted run | random, adaptive_uncertainty_diversity, entropy, class_group_balanced_entropy, badge |
| Budgets | 50, 100, 200, 300, 400 |
| Seeds | 13, 21, 34 |
| Initial seed size | 9 |
| Train cap | 500 |
| Test cap | 250 |
| Run id | 20260612-204850 |

### Metric Definitions

| Metric | Definition |
| --- | --- |
| macro-F1 | Unweighted mean F1 across classes; primary quality metric for imbalanced/many-class classification. |
| AULC macro-F1 | Area under the learning curve across labeled budgets; rewards strategies that perform well throughout the budget path, not only at the final point. |
| lift vs random | Strategy metric minus matched random metric for the same dataset, seed, and budget. Positive means active learning beat random sampling. |
| full-train reference | Same benchmark model trained on the entire capped training pool, outside the SDK acquisition loop. This is the parity reference. |
| label coverage fraction | Fraction of dataset labels/classes represented in the labeled training set by that budget. |
| zero-recall class fraction | Fraction of classes with zero recall on the test split at evaluation time. Lower is better. |
| ECE | Expected calibration error; lower indicates better confidence calibration. |
| NLL | Negative log likelihood; lower is better. |

## Quality Gate Result

- Result: PASS
- Metrics rows in promoted run: 150
- Selection differentiation: comparable groups `30`; collapsed selected-order groups `0`; collapsed effective-strategy groups `0`; pairwise collapsed strategy pairs `0`.
- Passing strategies: `adaptive_uncertainty_diversity, class_group_balanced_entropy, entropy`.
- Strategies with non-negative lift: `adaptive_uncertainty_diversity, badge, class_group_balanced_entropy, entropy`.

Important gate outcomes:

| Gate | Status | Detail |
| --- | --- | --- |
| `random_baseline_present` | PASS | Random baseline rows are required for strategy comparisons. |
| `at_least_one_non_random_strategy` | PASS | Quality gates need at least one strategy to compare against random. |
| `random_baseline_complete` | PASS | Every non-random dataset/seed/budget row must have a matching random baseline. |
| `single_strategy_quality_candidate` | PASS | At least one strategy must simultaneously have non-negative final macro-F1 lift, non-negative AULC lift, meaningful-acquisition non-loss-rate >= 2/3, final-budget non-loss-rate ... |
| `non_random_non_negative_quality_lift` | PASS | At least one non-random strategy must have a computable non-negative final macro-F1 or AULC lift versus matched random. |
| `win_rate_available` | PASS | At least one strategy-vs-random macro-F1 win-rate must be computable. |
| `non_random_strategy_differentiation` | PASS | When selections.csv is available, non-random strategies must not all share the same selected order in every comparable dataset/seed/budget group, and no strategy pair may be ide... |
| `runtime_summary_present` | PASS | Runtime summary rows with finite mean runtime must be present. |
| `manifest_evidence_category_present` | PASS | When manifest.json exists, the report must expose its evidence category. |
| `strategy_claim_categories_separated` | PASS | SDK-native, manual formula-reference, external formula-shim, and native external strategy claims must be categorized without overlap. |
| `stage11_real_standard_seed_count` | PASS | Stage 11 standard real evidence requires at least three distinct seeds; real_smoke is smoke-only and not standard evidence. |
| `stage11_real_standard_metrics_present` | PASS | Stage 11 standard real metrics.csv rows must include finite calibration, coverage, and zero-recall metrics. |
| `stage11_real_standard_full_train_calibration_present` | PASS | Stage 11 standard real full_train_reference.csv rows must include finite calibration metrics. |

## Full-Train Reference

This is the capped full-train comparator: the same benchmark model trained on all capped training examples, not through active learning.

| Dataset | Train size | Test size | Macro-F1 mean | Macro-F1 std | Accuracy mean | ECE mean | NLL mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| banking77 | 500 | 250 | 0.3191 | 0.0633 | 0.3893 | 0.3160 | 3.4162 |
| dair_ai_emotion | 500 | 250 | 0.1816 | 0.0458 | 0.4320 | 0.0831 | 1.4704 |

## Final-Budget Main Evidence

Positive `gap vs full train` means the active-learning run slightly exceeded the capped full-train reference; negative means it remained below it.

| Dataset | Budget | Strategy | Macro-F1 mean | Macro-F1 std | Lift vs random | Gap vs full train | N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| banking77 | 400 | adaptive_uncertainty_diversity | 0.3199 | 0.0259 | 0.0379 | 0.0008 | 3 |
| banking77 | 400 | badge | 0.2972 | 0.0420 | 0.0152 | -0.0220 | 3 |
| banking77 | 400 | class_group_balanced_entropy | 0.2911 | 0.0436 | 0.0091 | -0.0280 | 3 |
| banking77 | 400 | entropy | 0.3053 | 0.0400 | 0.0233 | -0.0138 | 3 |
| banking77 | 400 | random | 0.2820 | 0.0301 | n/a | -0.0371 | 3 |
| dair_ai_emotion | 400 | adaptive_uncertainty_diversity | 0.1744 | 0.0378 | 0.0017 | -0.0072 | 3 |
| dair_ai_emotion | 400 | badge | 0.1908 | 0.0239 | 0.0181 | 0.0092 | 3 |
| dair_ai_emotion | 400 | class_group_balanced_entropy | 0.1839 | 0.0341 | 0.0113 | 0.0023 | 3 |
| dair_ai_emotion | 400 | entropy | 0.1795 | 0.0386 | 0.0068 | -0.0021 | 3 |
| dair_ai_emotion | 400 | random | 0.1727 | 0.0305 | n/a | -0.0089 | 3 |

## Mid-Budget Efficiency Snapshot

Budget 200 is useful for slides because random already produces a real result, but active learning has room to show label-efficiency gains.

| Dataset | Strategy | Macro-F1 @200 | Lift vs random @200 | Share of full-train macro-F1 |
| --- | --- | --- | --- | --- |
| banking77 | adaptive_uncertainty_diversity | 0.1962 | 0.0493 | 0.6147 |
| banking77 | badge | 0.1804 | 0.0335 | 0.5651 |
| banking77 | class_group_balanced_entropy | 0.1668 | 0.0200 | 0.5228 |
| banking77 | entropy | 0.1554 | 0.0085 | 0.4868 |
| banking77 | random | 0.1468 | n/a | 0.4601 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 0.1656 | 0.0245 | 0.9118 |
| dair_ai_emotion | badge | 0.1590 | 0.0180 | 0.8758 |
| dair_ai_emotion | class_group_balanced_entropy | 0.1737 | 0.0326 | 0.9563 |
| dair_ai_emotion | entropy | 0.1716 | 0.0306 | 0.9450 |
| dair_ai_emotion | random | 0.1410 | n/a | 0.7767 |

## Main Learning-Curve Table

Mean and standard deviation are over seeds `13, 21, 34`.

| Dataset | Strategy | Budget | Macro-F1 mean | Macro-F1 std | N |
| --- | --- | --- | --- | --- | --- |
| banking77 | adaptive_uncertainty_diversity | 100 | 0.0768 | 0.0063 | 3 |
| banking77 | adaptive_uncertainty_diversity | 200 | 0.1962 | 0.0275 | 3 |
| banking77 | adaptive_uncertainty_diversity | 300 | 0.2451 | 0.0150 | 3 |
| banking77 | adaptive_uncertainty_diversity | 400 | 0.3199 | 0.0259 | 3 |
| banking77 | adaptive_uncertainty_diversity | 50 | 0.0457 | 0.0029 | 3 |
| banking77 | badge | 100 | 0.0691 | 0.0105 | 3 |
| banking77 | badge | 200 | 0.1804 | 0.0045 | 3 |
| banking77 | badge | 300 | 0.2330 | 0.0232 | 3 |
| banking77 | badge | 400 | 0.2972 | 0.0420 | 3 |
| banking77 | badge | 50 | 0.0285 | 0.0058 | 3 |
| banking77 | class_group_balanced_entropy | 100 | 0.0831 | 0.0112 | 3 |
| banking77 | class_group_balanced_entropy | 200 | 0.1668 | 0.0323 | 3 |
| banking77 | class_group_balanced_entropy | 300 | 0.2306 | 0.0272 | 3 |
| banking77 | class_group_balanced_entropy | 400 | 0.2911 | 0.0436 | 3 |
| banking77 | class_group_balanced_entropy | 50 | 0.0343 | 0.0070 | 3 |
| banking77 | entropy | 100 | 0.0975 | 0.0251 | 3 |
| banking77 | entropy | 200 | 0.1554 | 0.0204 | 3 |
| banking77 | entropy | 300 | 0.2591 | 0.0401 | 3 |
| banking77 | entropy | 400 | 0.3053 | 0.0400 | 3 |
| banking77 | entropy | 50 | 0.0520 | 0.0140 | 3 |
| banking77 | random | 100 | 0.0725 | 0.0198 | 3 |
| banking77 | random | 200 | 0.1468 | 0.0665 | 3 |
| banking77 | random | 300 | 0.2184 | 0.0557 | 3 |
| banking77 | random | 400 | 0.2820 | 0.0301 | 3 |
| banking77 | random | 50 | 0.0605 | 0.0285 | 3 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 100 | 0.1459 | 0.0135 | 3 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 200 | 0.1656 | 0.0409 | 3 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 300 | 0.1716 | 0.0409 | 3 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 400 | 0.1744 | 0.0378 | 3 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 50 | 0.1245 | 0.0199 | 3 |
| dair_ai_emotion | badge | 100 | 0.1472 | 0.0093 | 3 |
| dair_ai_emotion | badge | 200 | 0.1590 | 0.0539 | 3 |
| dair_ai_emotion | badge | 300 | 0.1730 | 0.0408 | 3 |
| dair_ai_emotion | badge | 400 | 0.1908 | 0.0239 | 3 |
| dair_ai_emotion | badge | 50 | 0.1113 | 0.0210 | 3 |
| dair_ai_emotion | class_group_balanced_entropy | 100 | 0.1487 | 0.0076 | 3 |
| dair_ai_emotion | class_group_balanced_entropy | 200 | 0.1737 | 0.0487 | 3 |
| dair_ai_emotion | class_group_balanced_entropy | 300 | 0.1926 | 0.0561 | 3 |
| dair_ai_emotion | class_group_balanced_entropy | 400 | 0.1839 | 0.0341 | 3 |
| dair_ai_emotion | class_group_balanced_entropy | 50 | 0.1482 | 0.0437 | 3 |
| dair_ai_emotion | entropy | 100 | 0.1451 | 0.0299 | 3 |
| dair_ai_emotion | entropy | 200 | 0.1716 | 0.0423 | 3 |
| dair_ai_emotion | entropy | 300 | 0.1753 | 0.0243 | 3 |
| dair_ai_emotion | entropy | 400 | 0.1795 | 0.0386 | 3 |
| dair_ai_emotion | entropy | 50 | 0.1303 | 0.0184 | 3 |
| dair_ai_emotion | random | 100 | 0.1371 | 0.0276 | 3 |
| dair_ai_emotion | random | 200 | 0.1410 | 0.0247 | 3 |
| dair_ai_emotion | random | 300 | 0.1577 | 0.0281 | 3 |
| dair_ai_emotion | random | 400 | 0.1727 | 0.0305 | 3 |
| dair_ai_emotion | random | 50 | 0.1361 | 0.0104 | 3 |

## Whole-Curve AULC Evidence

AULC summarizes the entire learning curve and protects against overclaiming from one final-budget point.

| Dataset | Strategy | AULC macro-F1 mean | AULC std | AULC lift vs random | N |
| --- | --- | --- | --- | --- | --- |
| banking77 | adaptive_uncertainty_diversity | 0.1915 | 0.0076 | 0.0270 | 3 |
| banking77 | badge | 0.1774 | 0.0138 | 0.0129 | 3 |
| banking77 | class_group_balanced_entropy | 0.1754 | 0.0253 | 0.0109 | 3 |
| banking77 | entropy | 0.1866 | 0.0226 | 0.0221 | 3 |
| banking77 | random | 0.1645 | 0.0394 | n/a | 3 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 0.1614 | 0.0329 | 0.0123 | 3 |
| dair_ai_emotion | badge | 0.1616 | 0.0322 | 0.0125 | 3 |
| dair_ai_emotion | class_group_balanced_entropy | 0.1734 | 0.0391 | 0.0243 | 3 |
| dair_ai_emotion | entropy | 0.1652 | 0.0277 | 0.0161 | 3 |
| dair_ai_emotion | random | 0.1491 | 0.0246 | n/a | 3 |

## Runtime Summary

Runtime is per evaluated curve point inside the benchmark harness; compare within this run, not as production serving latency.

| Dataset | Strategy | Runtime mean seconds | Runtime std | Rows |
| --- | --- | --- | --- | --- |
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

## Coverage And Zero-Recall Diagnostics At Final Budget

| Dataset | Strategy | Budget | Label coverage mean | Coverage std | Zero-recall class fraction mean | N |
| --- | --- | --- | --- | --- | --- | --- |
| banking77 | adaptive_uncertainty_diversity | 400 | 98.7% | 0.0000 | 40.1% | 3 |
| banking77 | badge | 400 | 99.1% | 0.0075 | 43.6% | 3 |
| banking77 | class_group_balanced_entropy | 400 | 99.6% | 0.0075 | 44.5% | 3 |
| banking77 | entropy | 400 | 99.6% | 0.0075 | 44.0% | 3 |
| banking77 | random | 400 | 99.1% | 0.0075 | 44.9% | 3 |
| dair_ai_emotion | adaptive_uncertainty_diversity | 400 | 100.0% | 0.0000 | 50.0% | 3 |
| dair_ai_emotion | badge | 400 | 100.0% | 0.0000 | 55.6% | 3 |
| dair_ai_emotion | class_group_balanced_entropy | 400 | 100.0% | 0.0000 | 50.0% | 3 |
| dair_ai_emotion | entropy | 400 | 100.0% | 0.0000 | 50.0% | 3 |
| dair_ai_emotion | random | 400 | 100.0% | 0.0000 | 55.6% | 3 |

## Calibration Diagnostics At Final Budget

| Dataset | Strategy | Brier mean | NLL mean | ECE mean | N |
| --- | --- | --- | --- | --- | --- |
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

## Slide Story Options

| Slide | Claim | Best visual | Evidence note |
| --- | --- | --- | --- |
| 1 | SDK can reach full-train quality with a capped labeling budget | Line curve + full-train reference marker | Banking77 adaptive at budget 400 equals capped full train within rounding; Emotion adaptive is close to full train. |
| 2 | Active learning matters most on many-class text classification | Banking77 learning curve | Adaptive has clear mid-budget and final-budget lift over random on 77-class Banking77. |
| 3 | Quality improvements are not from collapsed selections | Selection differentiation / coverage table | Quality gate reports 30 comparable groups and zero collapsed selected-order/effective-strategy groups. |
| 4 | Several SDK strategies beat random, with different tradeoffs | Final budget dot plot or grouped bar | Entropy/badge/adaptive all non-loss at final budget; class-balanced entropy has strongest overall win-rate but lower final lift in Banking77. |
| 5 | The evidence is rigorous but scoped | Methods/caveats slide | Current proof uses capped real datasets, 3 seeds, TF-IDF/logistic benchmark adapter; it validates acquisition behavior, not GPU/Transformer production parity. |

## Text-Only Chart Blueprints

| Chart | Recommended form | Source | Fields | Takeaway | Slide role |
| --- | --- | --- | --- | --- | --- |
| Main learning curves | Uncertainty & benchmark / multi-series line | ours_datasets_metrics.csv | dataset, strategy, budget -> mean macro_f1; std over seed | Show quality as labeled budget grows; use random dashed black and adaptive highlighted. | Presentation main evidence slide |
| Full-train parity plot | Comparison / dot plot with reference line | ours_datasets_metrics.csv + full_train_reference.csv | final budget macro_f1 by strategy vs full_train macro_f1 | Show whether capped active learning reaches full-train reference. | Best single proof slide |
| Budget-200 efficiency | Grouped bar or lollipop | ours_datasets_metrics.csv | macro_f1 at budget 200; lift_vs_random | Show early-budget improvement before full cap. | Argument for label-cost reduction |
| AULC lift | Bar chart with zero baseline | promoted metrics.csv | aulc_macro_f1 by dataset/strategy/seed | Compare whole-curve quality, not only final point. | Robustness slide |
| Runtime vs quality | Scatter or connected dot | promoted metrics.csv | mean runtime_seconds vs final macro_f1 | Show strategy tradeoff: quality gains versus compute cost. | Engineering tradeoff slide |
| Coverage and zero-recall | Grouped bars or small table | promoted metrics.csv | label_coverage_fraction and zero_recall_class_fraction at final budget | Show whether selected labels cover classes and reduce blind spots. | Trust/diagnostics slide |

## Suggested Slide Copy

Use short visible slide text and keep the details in speaker notes. Candidate copy:

| Use | Copy |
| --- | --- |
| Headline | SDK active learning reaches full-train quality on capped real benchmarks. |
| Subheadline | On Banking77, adaptive uncertainty-diversity matches the capped full-train macro-F1 at 400 labels and beats random sampling by +0.0379 macro-F1. |
| Speaker note | This is a capped-real benchmark: 500 train examples, 250 test examples, 3 seeds, macro-F1 primary. The claim is about SDK acquisition behavior using the benchmark sklearn adapter, not about Transformer production parity. |
| Caveat label | Evidence boundary: current adapter is CPU sklearn TF-IDF/logistic regression. CUDA would matter for a separate neural-model benchmark, not this run. |

## Limitations And Honest Boundaries

| Boundary | Implication |
| --- | --- |
| Model adapter | The current evidence uses scikit-learn TF-IDF/logistic regression, not DistilBERT or another neural model. |
| Dataset caps | Train/test caps make runs reproducible and affordable, but the reported full-train reference is capped full-train, not original full dataset training. |
| Seed count | Three seeds are enough for the project quality gate and visible std bands, but not for high-confidence statistical inference. |
| CUDA | CUDA is available on the machine, but the current SDK benchmark path is CPU-only. A CUDA/Transformer experiment should be presented as a separate benchmark family. |

## Validation Notes

| Check | Result |
| --- | --- |
| Promoted metrics rows | 150 |
| Prepared main metrics rows | 150 |
| Quality gate | PASS |
| Focused tests after benchmark edits | `uv run pytest tests/test_quality_gate_report.py tests/test_sdk_first_benchmark_embedding_diagnostics.py -q` -> 35 passed |

## Minimal Reproduction Commands

```powershell
uv run --extra benchmarks python benchmarks/sdk_first_benchmark.py --preset real_medium --datasets banking77,dair_ai_emotion --strategies random,adaptive_uncertainty_diversity,entropy,class_group_balanced_entropy,badge --budgets 50,100,200,300,400 --seeds 13,21,34 --initial-seed-size 9 --max-train-samples 500 --max-test-samples 250 --output-dir benchmarks/results/runtime/quality_gate_adaptive_budget_curve_20260612 --overwrite
uv run python benchmarks/quality_gate_report.py benchmarks/results/runtime/quality_gate_adaptive_budget_curve_20260612
```

