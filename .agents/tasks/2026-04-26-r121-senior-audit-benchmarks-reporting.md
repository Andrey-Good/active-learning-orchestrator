# 2026-04-26-r121 Senior Audit: Benchmarks And Reporting

## Context

The user requested a senior-level code review and acceptance audit of the active-learning SDK: correctness, real behavior under edge cases, code quality, hacks, garbage, stress tests, a written findings file, and benchmark evidence showing where the SDK is worse than analogs or direct/manual implementation.

This subtask covers benchmark validity, artifact quality, report claims, and existing benchmark audit tests.

## Goal

Validate that benchmark evidence is honest and reproducible, identify misleading claims or weak comparison methodology, and report whether current artifacts satisfy the user's request.

## Responsibility Boundaries

In scope:

- `benchmarks/audit_sdk_vs_manual.py`
- `benchmarks/results/audit_sdk_vs_manual_*`
- `tests/test_audit_benchmark_comparison.py`
- `docs/SDK_CODE_AUDIT_2026-04-26.md`
- `docs/SDK_CODE_AUDIT_REPEAT_2026-04-26.md`
- `README.md` claims related to quality/benchmarks

Out of scope:

- Editing files directly.
- Implementing real external adapters.
- Runtime or strategy implementation changes except as they affect benchmark claims.

## Explicit Prohibitions

- Do not modify or delete files.
- Do not revert existing uncommitted changes.
- Do not fabricate comparisons to external libraries.
- Do not present microbenchmark results as end-to-end active-learning quality.

## Execution Plan

1. Inspect benchmark harness and tests.
2. Run targeted benchmark tests and a smoke benchmark if feasible.
3. Compare generated claims against what the benchmark actually proves.
4. Return concise findings with artifact paths, commands, and recommended next benchmark work.

## Acceptance Criteria

- Clear verdict on benchmark/report acceptability.
- Honest distinction between manual-loop comparison, external analog status, and end-to-end quality.
- Reproduction commands and artifact paths are identified.

## Dependencies

Can run in parallel with runtime and strategy audit subtasks.
