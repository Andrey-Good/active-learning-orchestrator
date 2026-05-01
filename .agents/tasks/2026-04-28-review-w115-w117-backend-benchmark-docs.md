# Review W115-W117 Backend/Benchmark/Docs/API Fixes

## Context

Workers fixed Label Studio bare-list responses, mapping prelabel score validation, custom backend factory errors, quality-gate string metadata, sklearn mypy policy, strict audit benchmark probability validation, benchmark/docs evidence claims, docs index, and public API exports.

## Goal

Review only. Confirm correctness and identify regressions in:

- `src/active_learning_sdk/backends/label_studio.py`
- `src/active_learning_sdk/backends/base.py`
- `benchmarks/quality_gate_report.py`
- `benchmarks/audit_sdk_vs_manual.py`
- `src/active_learning_sdk/adapters/sklearn.py`
- `src/active_learning_sdk/__init__.py`
- README/docs evidence updates
- related tests

## In Scope

- Label Studio pagination normalization for dict/list responses.
- Direct prelabel validation consistency.
- Custom backend error contract.
- Quality-gate parser handling string metadata without corrupting numeric metrics.
- Mypy policy narrowness.
- Benchmark docs not overclaiming native external-library workflow evidence.
- Current backlog status accurately reflects remaining roadmap vs fixed executable defects.

## Out Of Scope

- Runtime state/cache internals except public API exports.
- Implementing native modAL/scikit-activeml workflow benchmarks.
- Large architecture refactors.

## Constraints

- Review only, do not edit.
- Provide concrete file/line findings.
- Severity P1/P2 only unless a P3 is very cheap and real.

## Suggested Validation

- `uv run pytest tests/test_current_open_audit_defects_2026_04_28.py tests/test_label_backends.py tests/test_audit_benchmark_comparison.py tests/test_quality_gate_report.py tests/test_deep_audit_public_api_packaging_2026_04_28.py -q`
- `uv run mypy src`
- `uv run --with ruff ruff check .`

## Acceptance Criteria

- State whether this scope is accepted.
- If rejected, list exact blockers and reproduction.
