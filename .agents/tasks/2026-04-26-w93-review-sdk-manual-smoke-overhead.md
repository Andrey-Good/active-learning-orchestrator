# W93 Review: SDK/manual smoke overhead reduction

## Context
The SDK/manual audit benchmark previously reported that the SDK uncertainty path was about 5x-6.6x slower than a direct manual formula loop on a tiny smoke workload while selecting the same IDs. A performance patch reduces avoidable overhead while preserving parity and runtime validation.

## Goal
Review the implemented performance changes for correctness, benchmark fairness, and regression risk.

## Responsibility Boundaries
This is a read-only review task. Do not edit files.

## In Scope
- `benchmarks/audit_sdk_vs_manual.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- `tests/test_audit_benchmark_comparison.py`
- relevant audit docs/results under `docs/` and `benchmarks/results/`

## Out of Scope
- Adding new strategies or benchmark datasets.
- Reworking broader SDK architecture.
- Changing test expectations.

## Special Attention
- Confirm the benchmark remains honest: SDK and manual paths should compare the same formula-level acquisition work, with any warmed cache behavior explicitly documented.
- Confirm selected IDs remain parity-equivalent.
- Confirm built-in strategy registration changes do not break custom strategy overrides or availability semantics.
- Confirm optimizations do not weaken probability validation.
- Confirm tie-breaking remains deterministic and stable.

## Forbidden Actions
- Do not revert unrelated user or agent changes.
- Do not edit files.
- Do not run destructive git commands.

## Review Plan
1. Inspect the relevant diff and current files.
2. Check whether the benchmark contract and caveats describe the warmed cache setup.
3. Check whether scheduler and uncertainty changes preserve behavior.
4. Report concrete findings with severity, file/line references, and suggested fixes.

## Acceptance Criteria
- No correctness regression in strategy selection, validation, or custom strategy behavior.
- Benchmark comparison is not misleading.
- Any remaining risk is clearly described.

## Dependencies
Runs after the W93 implementation patch.
