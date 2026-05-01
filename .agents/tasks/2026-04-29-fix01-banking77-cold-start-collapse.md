# Task: 2026-04-29-fix01-banking77-cold-start-collapse

## Context

Black-box stress found that the Banking77 capped real benchmark collapsed all non-random strategies to identical selections across seed/budget groups and they lost to random on final macro-F1.

Evidence:

- `.agents/tmp/blackbox_stress/FINAL_STRESS_REPORT.md`
- `.agents/tmp/blackbox_stress/wave2_real_model_findings.md`
- `.agents/tmp/blackbox_stress/reviews/wave2_real_model_review.md`
- `.agents/tmp/blackbox_stress/wave2_real_model/real_medium_banking77_strategy_matrix/`

## Goal

Identify why non-random strategies collapse in many-class cold-start capped real benchmarks and propose or implement the smallest safe SDK/benchmark fix that restores strategy differentiation without hiding weak quality evidence.

## Responsibility Boundaries

Explorer scope:

- Inspect benchmark artifacts, `benchmarks/sdk_first_benchmark.py`, strategy code, scheduler fallback behavior, and selected IDs/diagnostics.
- Determine whether collapse is caused by benchmark adapter limitations, missing embeddings, fallback policy, seed-label coverage, tie-breaking, strategy registry, or selection context.
- Recommend exact code areas and tests to change.

Out of scope for explorer:

- Do not edit files.
- Do not run long real benchmarks unless explicitly cheap and bounded.
- Do not claim universal quality improvements without evidence.

## Acceptance Criteria

- Written findings in `.agents/tmp/2026-04-29-fix01-banking77-cold-start-collapse-findings.md`.
- Explain root cause with evidence from artifacts/code.
- Recommend focused regression tests and a minimal fix.
- If the issue is not fully fixable quickly, separate release-blocking behavior from benchmark-quality limitation.

## Expected Validation

Prefer reading existing artifacts first. If running code, keep it small and under a few minutes.
