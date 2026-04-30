# Review Stage 3B: SelectionContext Extraction

## Context

Stage 3B extracted SelectionContext probability/embedding validation into `src/active_learning_sdk/runtime/selection_context_validation.py`.

## Goal

Review only. Confirm behavior is preserved and extraction did not introduce cache validation regressions.

## In Scope

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/runtime/selection_context_validation.py`
- `src/active_learning_sdk/runtime/__init__.py`
- `tests/test_core_refactor_characterization.py`

## Review Questions

- Are exceptions and messages compatible with existing tests/contracts?
- Does prediction cache invalid row eviction still work?
- Does embedding cache invalid row eviction still work?
- Did the extraction avoid public API changes?
- Are helper functions cohesive and not leaking too much engine state?

## Constraints

- Review only, do not edit.
- Provide concrete P1/P2 findings if any.

## Suggested Validation

- `uv run pytest tests/test_core_refactor_characterization.py tests/test_strategy_capabilities.py tests/test_w97_senior_audit_acceptance.py -q`
- `uv run pytest -q`

## Acceptance Criteria

- No P1/P2 findings.
- Stage 3B can be treated as complete before Stage 3C starts.
