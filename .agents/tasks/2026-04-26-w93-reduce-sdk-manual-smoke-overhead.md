# w93 - Reduce SDK vs Manual Smoke Overhead

## Context

The audit benchmark reports full selection parity for `entropy`, `margin`, and `least_confidence`, but SDK selection is about 5x-6x slower than the direct manual formula on a tiny frozen-probability workload.

## Goal

Reduce the measured SDK overhead without changing selected IDs or weakening runtime validation contracts.

## Ownership

Own:

- `benchmarks/audit_sdk_vs_manual.py`
- `tests/test_audit_benchmark_comparison.py`
- narrow hot-path changes in `src/active_learning_sdk/engine.py` and `src/active_learning_sdk/strategies/uncertainty.py` if justified by profiling

Do not change active-learning quality benchmarks, Label Studio backends, or unrelated SDK APIs.

## In Scope

- Measure current overhead and identify whether benchmark setup or SDK hot path dominates.
- Reuse stable scheduler/context objects inside the microbenchmark if that better represents selection-loop overhead.
- Optimize obvious hot-path costs that do not reduce safety:
  - avoid rebuilding scheduler registry when not needed;
  - avoid repeated built-in registry reconstruction;
  - keep deterministic tie-breaking and output validation.
- Add an acceptance test around overhead if it can be robust enough for CI. Prefer a loose threshold to avoid flaky timing.

## Out Of Scope

- No fake parity shortcuts.
- No disabling validation for production code just to win a microbenchmark.
- No claims against external libraries unless those comparisons actually run.

## Acceptance Criteria

- `uv run python benchmarks/audit_sdk_vs_manual.py --output-dir benchmarks/results/audit_sdk_vs_manual_smoke --budget 5 --repeats 200` succeeds.
- SDK/manual selected order remains identical for all supported strategies.
- Measured overhead ratio is materially lower than the reported 5x-6x range.
- `uv run pytest tests/test_audit_benchmark_comparison.py -q` passes.
- Full `uv run pytest -q` passes after integration.
