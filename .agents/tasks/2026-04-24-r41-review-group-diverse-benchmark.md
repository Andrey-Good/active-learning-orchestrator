# R41 - Review Group-Diverse Benchmark Results

## Relation To Overall Task
W31 added `group_diverse_entropy` to the benchmark harness and ran a targeted benchmark. The hypothesis result was mixed: group concentration improved, rare recall preserved, but macro-F1 stayed below random/mix on grouped duplicates.

## Goal
Read-only review of W31 benchmark integration and analysis. Check if the result interpretation is sound and whether artifacts support the stated conclusion.

## Scope
Inspect:
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md`
- `benchmarks/results/group_diverse_entropy/**`

Do not edit files.

## Acceptance Criteria
- Strategy is correctly represented in benchmark configs/artifacts.
- Analysis conclusion follows from metrics.
- No blocking issues remain before choosing next hypothesis.

## Expected Validation
Optional:
- strict JSON parse
- spot-check CSV summaries
