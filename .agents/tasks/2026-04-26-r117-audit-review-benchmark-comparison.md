# r117 - Review Benchmark Comparison Audit

## Context

This is the reviewer pass for `w93 - Audit Benchmark Comparison`.

## Goal

Review the benchmark comparison harness, tests, and benchmark-validity findings.

## Responsibility Boundaries

Read only:

- `benchmarks/audit_sdk_vs_manual.py`
- `tests/test_audit_benchmark_comparison.py`
- related benchmark files referenced by the worker
- the worker final response if provided by the orchestrator

Do not edit files.

## In Scope

- Confirm the manual-vs-SDK comparison is fair enough for an audit smoke benchmark.
- Check whether benchmark output semantics are honest and not overclaimed.
- Identify missing caveats or invalid comparisons.
- Check that tests verify meaningful behavior.

## Out Of Scope

- Runtime and strategy stress tests.
- Fixing code or tests.

## Acceptance Criteria

- Reviewer final response lists accepted findings and rejected/weak findings.
- Any requested changes are concrete and scoped.

## Dependencies

Run after `w93`.

## Parallel/Sequential Execution

Sequential after the benchmark worker.
