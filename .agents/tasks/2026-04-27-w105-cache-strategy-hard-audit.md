# Task W105: Cache/Strategy Hard-Audit Fixes

## Context

The hard senior audit added known-defect tests in `tests/test_hard_audit_known_defects_2026_04_27.py`. The cache/strategy subset covers cache capacity behavior and probability/label-schema contract validation.

## Goal

Fix cache and probability-backed strategy production behavior without editing tests or docs.

## Ownership

You may edit:

- `src/active_learning_sdk/cache.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- other strategy modules only if necessary for shared probability-width validation

Do not edit:

- `src/active_learning_sdk/engine.py`
- tests
- docs
- README

## Required Fixes

1. `InMemoryCacheStore(max_items=0)` must not crash with `StopIteration`; either no-op zero capacity or reject clearly. Prefer no-op if it preserves existing config ergonomics.
2. Probability-backed uncertainty strategies must reject probability width mismatches against `LabelSchema.labels` when the context exposes a label schema.
3. Review whether this validation should be centralized so `entropy`, `margin`, `least_confidence`, and class/group variants behave consistently.

## Constraints

- Keep deterministic tie-breaking and previous strict probability validation.
- Raise `ConfigurationError`, not raw Python exceptions, for invalid probability contracts.
- Do not loosen probability validation or reintroduce normalization of count-like rows.

## Acceptance Criteria

- The relevant hard-audit tests should be capable of passing once xfail markers are removed.
- Existing strategy tests should remain green.
- Return a concise summary of changed production behavior and any residual risks.
