# Current Benchmark Report

Updated with the 2026-06-12 adaptive budget-curve standard replay on
Banking77 and DAIR.AI Emotion.

This report promotes the current small SDK-first smoke and the current
dual-dataset capped-real standard replay as proof that the benchmark harness,
quality gates, runtime summary, manifest metadata, claim categories, full-train
references, and active-learning budget curves work together. Larger Stage 9
artifacts remain historical diagnostic evidence.

## Commands Run For This Report

```powershell
uv run pytest tests/test_quality_gate_report.py -q
uv run python benchmarks/sdk_first_benchmark.py --preset smoke --output-dir benchmarks/results/stage2_smoke_current --overwrite
uv run python benchmarks/quality_gate_report.py benchmarks/results/stage2_smoke_current
uv run --extra benchmarks python benchmarks/sdk_first_benchmark.py --preset real_medium --datasets banking77,dair_ai_emotion --strategies random,adaptive_uncertainty_diversity,entropy,class_group_balanced_entropy,badge --budgets 50,100,200,300,400 --seeds 13,21,34 --initial-seed-size 9 --max-train-samples 500 --max-test-samples 250 --output-dir benchmarks/results/runtime/quality_gate_adaptive_budget_curve_20260612 --overwrite
uv run python benchmarks/quality_gate_report.py benchmarks/results/runtime/quality_gate_adaptive_budget_curve_20260612
```

Results:

- `tests/test_quality_gate_report.py`: `8 passed`.
- Smoke benchmark output: `benchmarks/results/stage2_smoke_current/`.
- Smoke benchmark rows: `30` metrics rows, `30` selection rows, `30` stop-policy rows, `2` full-train reference rows.
- Smoke manifest: run id `20260428-204736`, preset `smoke`, artifact schema version `1`, git dirty `true`.
- Quality gate report: `PASS`, schema version `2`, evidence category `sdk_native_synthetic_diagnostic`.
- Real-medium benchmark output: `benchmarks/results/runtime/quality_gate_adaptive_budget_curve_20260612/`.
- Real-medium benchmark rows: `150` metrics rows, `150` selection rows, `90` stop-policy rows, `6` full-train reference rows.
- Real-medium quality gate report: `PASS`, schema version `2`, evidence category `sdk_native_capped_real_dataset`.

## Promoted 2026-06-12 Standard Real Evidence

Artifact: `benchmarks/results/runtime/quality_gate_adaptive_budget_curve_20260612/quality_gate.json`

Slide-prep dossier: `benchmarks/results/sdk_metrics_slide_dossier_20260612.md`

Configuration:

- Datasets: `banking77`, `dair_ai_emotion`
- Seeds: `13`, `21`, `34`
- Budgets: `50`, `100`, `200`, `300`, `400`
- Initial seed size: `9`
- Train cap: `500`
- Test cap: `250`
- Strategies: `random`, `adaptive_uncertainty_diversity`, `entropy`, `class_group_balanced_entropy`, `badge`
- Claim category: SDK-native capped-real diagnostic evidence using the benchmark-owned sklearn text adapter.

Quality-gate checks: all Stage 11 standard real-data checks pass, including
three-seed coverage, calibration metrics, full-train reference calibration,
random-baseline completeness, non-random strategy differentiation, runtime
summary, and separated strategy claim categories.

Passing strategies under the combined gate:

- `adaptive_uncertainty_diversity`
- `class_group_balanced_entropy`
- `entropy`

Headline macro-F1 results at the final budget:

| Dataset | Full-train reference | Random @400 | SDK adaptive @400 | SDK adaptive minus full | SDK adaptive lift vs random |
| --- | ---: | ---: | ---: | ---: | ---: |
| `banking77` | `0.3191` | `0.2820` | `0.3199` | `+0.0008` | `+0.0379` |
| `dair_ai_emotion` | `0.1816` | `0.1727` | `0.1744` | `-0.0072` | `+0.0017` |

Mid-budget macro-F1 lift versus random:

| Dataset | Budget | Random | SDK adaptive | Lift |
| --- | ---: | ---: | ---: | ---: |
| `banking77` | `200` | `0.1468` | `0.1962` | `+0.0493` |
| `dair_ai_emotion` | `200` | `0.1410` | `0.1656` | `+0.0245` |

Overall quality-gate summaries for `adaptive_uncertainty_diversity`:

- Final-budget macro-F1 lift vs random: `+0.0379` on Banking77 and `+0.0017` on DAIR.AI Emotion.
- Mean macro-F1 AULC lift vs random: `+0.0196`.
- Meaningful-acquisition non-loss rate vs random: `0.6667`.
- Final-budget non-loss rate vs random: `0.8333`.

## Promoted Stage 2C Quality Gate

Artifact: `benchmarks/results/stage2_smoke_current/quality_gate.json`

Configuration:

- Datasets: `separable_topics`, `rare_class_trap`
- Seed: `13`
- Budgets: `12`, `24`, `36`
- Strategies: `random`, `entropy`, `margin`, `least_confidence`, `mix_entropy_random`
- Claim category: SDK-native synthetic diagnostic evidence. No manual formula rows, external formula shims, or native external-library workflow rows are present in this smoke.

Quality-gate checks:

| Gate | Result |
| --- | --- |
| Random baseline present | PASS |
| At least one non-random strategy present | PASS |
| Random baseline complete for all comparisons | PASS |
| Single quality candidate passes final/AULC/non-loss criteria | PASS |
| At least one non-random strategy has non-negative lift | PASS |
| Win-rate is computable | PASS |
| Runtime summary is present | PASS |
| Manifest evidence category is present | PASS |
| Strategy claim categories are separated | PASS |

Passing strategies under the stricter combined gate:

- `least_confidence`
- `margin`
- `mix_entropy_random`

Strategies with any non-negative final macro-F1 or AULC lift:

- `entropy`
- `least_confidence`
- `margin`
- `mix_entropy_random`

## Stage 2C Smoke Metrics

Final-budget macro-F1 lift versus matched random:

| Dataset | Strategy | Mean lift |
| --- | --- | ---: |
| `rare_class_trap` | `entropy` | `+0.3805` |
| `rare_class_trap` | `least_confidence` | `+0.3805` |
| `rare_class_trap` | `margin` | `+0.3805` |
| `rare_class_trap` | `mix_entropy_random` | `+0.3673` |
| `separable_topics` | `entropy` | `-0.0278` |
| `separable_topics` | `least_confidence` | `+0.0000` |
| `separable_topics` | `margin` | `+0.0000` |
| `separable_topics` | `mix_entropy_random` | `+0.0000` |

Mean macro-F1 AULC lift versus matched random:

| Strategy | Mean AULC lift | Comparisons |
| --- | ---: | ---: |
| `entropy` | `+0.1783` | `2` |
| `least_confidence` | `+0.1675` | `2` |
| `margin` | `+0.1715` | `2` |
| `mix_entropy_random` | `+0.1871` | `2` |

Win/non-loss rates:

| Strategy | Win rate | Non-loss rate | Comparisons |
| --- | ---: | ---: | ---: |
| `entropy` | `0.6667` | `0.6667` | `6` |
| `least_confidence` | `0.6667` | `0.8333` | `6` |
| `margin` | `0.6667` | `0.8333` | `6` |
| `mix_entropy_random` | `0.6667` | `1.0000` | `6` |

Mean runtime summary:

| Dataset | Strategy | Mean runtime seconds |
| --- | --- | ---: |
| `rare_class_trap` | `entropy` | `0.0261` |
| `rare_class_trap` | `least_confidence` | `0.0265` |
| `rare_class_trap` | `margin` | `0.0263` |
| `rare_class_trap` | `mix_entropy_random` | `0.0271` |
| `rare_class_trap` | `random` | `0.0215` |
| `separable_topics` | `entropy` | `0.0223` |
| `separable_topics` | `least_confidence` | `0.0252` |
| `separable_topics` | `margin` | `0.0230` |
| `separable_topics` | `mix_entropy_random` | `0.0219` |
| `separable_topics` | `random` | `0.0175` |

## Retained Diagnostic Evidence

These artifacts mix fresh 2026-06-12 evidence with older retained diagnostics.
The dual-dataset 2026-06-12 row is the current capped-real standard replay;
older two-seed or single-dataset rows must not be cited as current standard
evidence.

- `benchmarks/results/runtime/local_gate_synthetic_v1`: PASS across `separable_topics`, `rare_class_trap`, and `grouped_duplicates` with budgets `16,32,48,64,96` and seeds `13,21,34`.
- `benchmarks/results/runtime/quality_gate_adaptive_budget_curve_20260612`: PASS on current capped-real standard evidence for Banking77 and DAIR.AI Emotion with budgets `50,100,200,300,400`, seeds `13,21,34`, train cap `500`, test cap `250`; `adaptive_uncertainty_diversity` final macro-F1 lift vs random `+0.0379` on Banking77 and `+0.0017` on DAIR.AI Emotion, mean AULC lift `+0.0196`, meaningful-acquisition non-loss rate `0.6667`, final-budget non-loss rate `0.8333`.
- `benchmarks/results/runtime/quality_gate_banking77_wave6_current`: historical capped Banking77 standard evidence with budgets `50,100,200`, seeds `13,21,34`, train cap `500`, test cap `250`; superseded by the dual-dataset 2026-06-12 standard replay.
- `benchmarks/results/runtime/quality_gate_banking77_budget500_v1`: historical two-seed capped Banking77 diagnostic only; not current standard evidence because the current `real_medium` contract requires at least three seeds.
- `benchmarks/results/runtime/quality_gate_emotion_adaptive_v2`: historical capped DAIR.AI Emotion evidence with budgets `50,100,200`, seeds `13,21,34`, train cap `300`, test cap `300`; superseded by the dual-dataset 2026-06-12 standard replay.
- `benchmarks/results/stage9_final`: legacy SDK-first synthetic diagnostic run with `1,440` metrics rows, `1,440` selection rows, and `864` stop-policy rows.
- `benchmarks/results/stage9_reference`: legacy reference/formula diagnostic run with `495` metrics rows, `495` selection rows, and `180` formula-equivalence rows.

Stage 9 retained headline numbers:

- Best overall mean macro-F1 AULC in the Stage 9 synthetic diagnostic matrix: `class_group_balanced_entropy` at `0.996018`.
- Random mean macro-F1 AULC: `0.948852`; best retained Stage 9 delta vs random: `+0.047166`.
- Formula-equivalence diagnostics: mean Jaccard `0.985537`, min Jaccard `0.684211`, exact selected order `139/180`, and `0.000000` macro-F1 AULC diffs for equivalent SDK/manual formula pairs.

## Claim Boundaries And Evidence Gaps

- The promoted Stage 2C smoke is a small synthetic gate, not a large benchmark rerun.
- Banking77 and DAIR.AI Emotion have a fresh current capped-real standard replay with three seeds.
- Stage 2C did not run native external-library workflow benchmarks. Formula-shim or manual-reference evidence must not be cited as native `modAL` or `skactiveml` workflow evidence.
- The benchmark adapter is still a benchmark-only scikit-learn TF-IDF/logistic-regression adapter.
- These results support controlled SDK validation and quality-gate health, not broad real-world production superiority.
