# R72 - Final Stage 1 System Review

## Context
Stage 1 goal: capability contracts and adapter baseline.

Completed subtasks:
- W52 implemented strategy capability declarations and validation.
- R68 found a protocol-stub capability bug.
- W54 fixed protocol-stub detection.
- W53 implemented `SklearnTextClassifierAdapter`.
- R69 found probability validation bugs.
- W55 fixed probability validation.
- R70 reviewed the fix-loop with no findings.
- W56 updated project smoke to use the public sklearn adapter.
- R71 reviewed project smoke with no findings.

## Goal
Perform the final end-to-end Stage 1 review and decide whether Stage 1 is safe to close.

## Responsibility Boundaries
- This is a read-only system review.
- Review Stage 1 as an integrated product slice.

## In Scope
- Capability contracts:
  - `src/active_learning_sdk/adapters/base.py`
  - `src/active_learning_sdk/strategies/base.py`
  - `src/active_learning_sdk/strategies/uncertainty.py`
  - `src/active_learning_sdk/engine.py`
  - `tests/test_strategy_capabilities.py`
- Sklearn adapter:
  - `src/active_learning_sdk/adapters/sklearn.py`
  - `src/active_learning_sdk/adapters/__init__.py`
  - `tests/test_sklearn_adapter.py`
- Project smoke:
  - `benchmarks/sdk_first_benchmark.py`
  - `tests/test_project_smoke_benchmark.py`
- Public exports and import smoke.

## Out of Scope
- Do not edit files.
- Do not implement Stage 2.
- Do not regenerate committed benchmark artifacts.
- Do not review unrelated pre-existing dirty files.

## Review Questions
- Do all built-in strategies declare correct capability requirements?
- Does configure/attach fail fast for missing strategy capabilities?
- Is the protocol-stub bug fixed?
- Is `coreset_kcenter` still rejected as unsupported in this build?
- Is `SklearnTextClassifierAdapter` public, deterministic, and strict about malformed outputs?
- Does project smoke use the public adapter and a public SDK loop?
- Are tests sufficient for Stage 1 exit criteria?

## Validation
- `uv run --group dev pytest -q`
- `uv run python benchmarks/sdk_first_benchmark.py --preset project_smoke --output-dir <temp path>`
- `uv run python - <<'PY'` import smoke for `SklearnTextClassifierAdapter`, `inspect_model_capabilities`, and built-in strategy names if convenient.

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No open blockers remain for Stage 1.
- Reviewer explicitly says Stage 1 can close if clean.
