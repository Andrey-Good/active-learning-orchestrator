# Stage 10B: Sklearn Adapter Public Readiness Audit

## Task Identifier

stage10b-sklearn-adapter-audit

## Context

The scikit-learn text adapter is the first concrete adapter users can run without
benchmark-only fixtures. Stage 10 needs it to be honest, robust, dependency-light
when unused, and documented within its intended scope.

## Goal

Audit `SklearnTextClassifierAdapter` for production-preview readiness and list
required fixes.

## Responsibility Boundaries

In scope:

- estimator fit/predict/evaluate behavior;
- probability validation and class-order handling;
- `decision_function` fallback correctness;
- `get_model_id` cache/version semantics;
- optional import behavior and packaging/docs alignment;
- tests that prove adapter behavior on edge cases.

Out of scope:

- Implementing fixes.
- Modifying benchmark code.
- Adding non-sklearn adapters.

## Files May Be Read

- `src/active_learning_sdk/adapters/sklearn.py`
- `src/active_learning_sdk/adapters/__init__.py`
- `tests/test_sklearn_adapter.py`
- `README.md`
- `pyproject.toml`
- related docs/tests if needed

## Files May Be Changed

- Only `.agents/tmp/2026-04-28-stage10b-sklearn-adapter-audit.md`

## Files Must Not Be Touched

- Production source files.
- Public docs outside the tmp report.
- Tests.

## Architectural Constraints

- Adapter should wrap estimator contract failures as `ModelAdapterError`.
- Adapter should not import sklearn from root package import.
- Adapter must not normalize invalid probability rows silently.
- Adapter should be deterministic for docs and smoke workloads.

## Special Attention

- Binary `decision_function` class order may be easy to get wrong.
- One-class fit errors should be clear.
- Empty inputs should not force fitted checks unnecessarily unless contract says so.
- Batch size coercion should not hide harmful caller bugs if it affects behavior.

## Forbidden Actions

- Do not edit code.
- Do not invent benchmark claims.
- Do not weaken adapter validation.

## Execution Plan

1. Inspect sklearn adapter implementation and tests.
2. Identify missing edge-case coverage.
3. Confirm whether the adapter can serve as the documented public example.
4. Write a severity-ranked audit report.

## Acceptance Criteria

- Report has an accept/reject verdict.
- Any P1/P2 finding includes a concrete failing scenario.
- Suggested fixes avoid broad rewrites unless justified.

## Expected Validations

- Optional focused test run is allowed, but keep this primarily read-only.

## Dependencies

- None.

## Parallelism

Can run in parallel with Stage 10A and Stage 10C.
