# Stage 3B: SelectionContext Cache Validation Extraction

## Context

`SelectionContext.predict_proba` and `SelectionContext.embed` are D-complexity hotspots. Stage 3B should reduce complexity by extracting validation/key helpers while preserving behavior.

## Dependencies

Run after Stage 3A characterization tests exist.

## Goal

Refactor selection context prediction/embedding cache validation into a small internal module without changing public API or behavior.

## Ownership

You may edit:

- `src/active_learning_sdk/engine.py`
- new internal module under `src/active_learning_sdk/runtime/` or similar
- focused tests only if needed

Do not edit strategies, backends, configs, benchmarks, or docs.

## In Scope

- Extract probability row validation and embedding row validation helpers.
- Extract cache-key/scoping helpers if useful.
- Keep `SelectionContext` public methods and exceptions behavior stable.
- Keep root public API unchanged.

## Constraints

- No behavior changes.
- No broad module split.
- Run characterization tests and full suite.

## Suggested Validation

- `uv run pytest tests/test_core_refactor_characterization.py tests/test_strategy_capabilities.py tests/test_w97_senior_audit_acceptance.py -q`
- `uv run pytest -q`
- `uv run --with radon radon cc src\active_learning_sdk\engine.py -s`

## Acceptance Criteria

- Full suite remains green.
- `SelectionContext.predict_proba` and/or `embed` complexity decreases.
