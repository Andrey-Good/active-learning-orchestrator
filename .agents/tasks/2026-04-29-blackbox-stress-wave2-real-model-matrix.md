# Task: blackbox-stress-wave2-real-model-matrix

## Context
Second wave of the user's aggressive black-box SDK stress test. Source-code inspection is forbidden; use docs and public benchmark/runtime behavior only.

## Goal
Push real-data and model/strategy quality harder than wave 1: more seeds where feasible, multiple real datasets if network/dependencies allow, tiny budgets, strategy families that should behave differently, and metric/regression analysis versus random.

## Responsibility Boundaries
Owns only artifacts under `.agents/tmp/blackbox_stress/wave2_real_model/` and findings in `.agents/tmp/blackbox_stress/wave2_real_model_findings.md`.

## In Scope
- Read public docs and benchmark docs.
- Run documented benchmark commands only; do not inspect benchmark source.
- Use real datasets such as Banking77, DAIR.AI Emotion, or CLINC if accessible.
- Use bounded caps to finish locally.
- Analyze metrics, calibration, selected-order differentiation, zero-recall, non-loss rates, runtime, and quality gate results.
- Attempt at least one external/downloaded dataset or model path if feasible.

## Out of Scope
- Reading `src/**` or benchmark implementation code.
- Modifying SDK or benchmark source.
- Claiming standard-real proof from smoke-only artifacts.

## Must Not Touch
- `src/**`
- benchmark source files
- promoted benchmark results
- other stress worker directories

## Execution Plan
1. Run one multi-seed capped real-data benchmark into owned output.
2. Run one tiny-budget adversarial real or synthetic/real mixed benchmark.
3. Parse artifacts for low metrics, identical selections, calibration failures, and runtime cliffs.
4. Summarize skips and evidence limitations honestly.

## Acceptance Criteria
- At least 2 benchmark commands attempted.
- At least 2 datasets attempted if dependencies/network allow.
- At least 4 strategy families attempted.
- Findings include artifact paths and metric tables.
