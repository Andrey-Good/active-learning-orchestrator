# Current Benchmark Report

Updated with Wave6 Banking77 standard replay on 2026-04-29.

This report promotes the current small SDK-first smoke and its quality-gate
report as the current proof that the benchmark harness, quality gates, runtime
summary, manifest metadata, and claim categories work together. Larger Stage 9
artifacts remain historical diagnostic evidence. The Banking77 capped-real
standard gate below was freshly rerun after the Wave6 fixes.

## Commands Run For This Report

```powershell
uv run pytest tests/test_quality_gate_report.py -q
uv run python benchmarks/sdk_first_benchmark.py --preset smoke --output-dir benchmarks/results/stage2_smoke_current --overwrite
uv run python benchmarks/quality_gate_report.py benchmarks/results/stage2_smoke_current
```

Results:

- `tests/test_quality_gate_report.py`: `8 passed`.
- Smoke benchmark output: `benchmarks/results/stage2_smoke_current/`.
- Smoke benchmark rows: `30` metrics rows, `30` selection rows, `30` stop-policy rows, `2` full-train reference rows.
- Smoke manifest: run id `20260428-204736`, preset `smoke`, artifact schema version `1`, git dirty `true`.
- Quality gate report: `PASS`, schema version `2`, evidence category `sdk_native_synthetic_diagnostic`.

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

These artifacts mix fresh Wave6 evidence with older retained diagnostics. The
Banking77 Wave6 row is the current standard replay; older two-seed Banking77
rows must not be cited as current standard evidence.

- `benchmarks/results/runtime/local_gate_synthetic_v1`: PASS across `separable_topics`, `rare_class_trap`, and `grouped_duplicates` with budgets `16,32,48,64,96` and seeds `13,21,34`.
- `benchmarks/results/runtime/quality_gate_banking77_wave6_current`: PASS on current capped Banking77 standard evidence with budgets `50,100,200`, seeds `13,21,34`, train cap `500`, test cap `250`; `adaptive_uncertainty_diversity` final macro-F1 lift vs random `+0.0264`, AULC lift `+0.0088`, all-budget non-loss rate `0.8889`, final-budget non-loss rate `0.6667`.
- `benchmarks/results/runtime/quality_gate_banking77_budget500_v1`: historical two-seed capped Banking77 diagnostic only; not current standard evidence because the current `real_medium` contract requires at least three seeds.
- `benchmarks/results/runtime/quality_gate_emotion_adaptive_v2`: PASS on capped DAIR.AI Emotion with budgets `50,100,200`, seeds `13,21,34`, train cap `300`, test cap `300`; `adaptive_uncertainty_diversity` final macro-F1 lift vs random `+0.0273`, AULC lift `+0.0187`, win/non-loss rate `0.8889`.
- `benchmarks/results/stage9_final`: legacy SDK-first synthetic diagnostic run with `1,440` metrics rows, `1,440` selection rows, and `864` stop-policy rows.
- `benchmarks/results/stage9_reference`: legacy reference/formula diagnostic run with `495` metrics rows, `495` selection rows, and `180` formula-equivalence rows.

Stage 9 retained headline numbers:

- Best overall mean macro-F1 AULC in the Stage 9 synthetic diagnostic matrix: `class_group_balanced_entropy` at `0.996018`.
- Random mean macro-F1 AULC: `0.948852`; best retained Stage 9 delta vs random: `+0.047166`.
- Formula-equivalence diagnostics: mean Jaccard `0.985537`, min Jaccard `0.684211`, exact selected order `139/180`, and `0.000000` macro-F1 AULC diffs for equivalent SDK/manual formula pairs.

## Claim Boundaries And Evidence Gaps

- The promoted Stage 2C smoke is a small synthetic gate, not a large benchmark rerun.
- Banking77 has a fresh current capped-real standard replay; DAIR.AI Emotion remains retained capped-real evidence until rerun.
- Stage 2C did not run native external-library workflow benchmarks. Formula-shim or manual-reference evidence must not be cited as native `modAL` or `skactiveml` workflow evidence.
- The benchmark adapter is still a benchmark-only scikit-learn TF-IDF/logistic-regression adapter.
- These results support controlled SDK validation and quality-gate health, not broad real-world production superiority.
