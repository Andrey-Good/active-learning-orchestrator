# Stage 3C: Split Validation Extraction

## Context

`_validate_persisted_splits` and `_resolve_splits` are D-complexity hotspots and have been a source of correctness bugs. After characterization tests, extract split validation into a dedicated internal module.

## Dependencies

Run after Stage 3A.

## Goal

Move split resolution/validation helpers out of `engine.py` into a focused internal module while preserving all public behavior.

## Ownership

You may edit:

- `src/active_learning_sdk/engine.py`
- new internal module under `src/active_learning_sdk/runtime/` or similar
- focused tests only if needed

Do not edit scheduler, caches, label backends, benchmarks, or docs.

## In Scope

- Extract explicit split validation.
- Extract column split resolution if low-risk.
- Extract persisted split stability checks if low-risk.
- Preserve exception types/messages enough for tests.

## Constraints

- No public API changes.
- No behavior changes.
- Keep changes incremental.

## Suggested Validation

- `uv run pytest tests/test_core_refactor_characterization.py tests/test_acceptance_public_contract_2026_04_27.py tests/test_hard_audit_known_defects_2026_04_27.py -q`
- `uv run pytest -q`
- `uv run --with radon radon cc src\active_learning_sdk\engine.py -s`

## Acceptance Criteria

- Full suite remains green.
- Split-related engine function complexity decreases.
