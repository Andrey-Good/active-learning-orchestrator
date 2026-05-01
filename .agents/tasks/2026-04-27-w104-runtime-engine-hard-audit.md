# Task W104: Runtime/Engine Hard-Audit Fixes

## Context

The hard senior audit in `docs/SENIOR_SDK_HARD_AUDIT_2026-04-27.md` added strict xfail tests in `tests/test_hard_audit_known_defects_2026_04_27.py`. The runtime/engine subset blocks production signoff.

## Goal

Fix runtime/engine production behavior without editing tests or docs.

## Ownership

You may edit:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/state/store.py` only if a durable state marker is clearly needed

Do not edit:

- tests
- docs
- README
- cache.py
- strategy modules

## Required Fixes

1. Selection pool must not include validation/test samples by default. Acquisition should be train-split only unless an explicit future policy exists.
2. `attach_runtime()` must detect `SplitConfig(mode="column")` split-assignment drift when IDs/texts do not change.
3. Cache-disabled `SelectionContext.predict_proba()` must share the same strict validation and cleaned row return behavior as the cache-enabled path.
4. `status()["active_round"]` must be `None` once the last round is complete.
5. If straightforward, make seed-label training durable/idempotent across restart before first selection.

## Constraints

- Preserve public APIs unless a hard-audit test proves the API contract is wrong.
- Raise existing SDK exceptions such as `ConfigurationError` or `DatasetMismatchError`, not raw Python errors.
- Avoid broad rewrites. Keep changes local and auditable.
- Do not remove safety validation that prior audits added.

## Acceptance Criteria

- The relevant hard-audit tests should be capable of passing once xfail markers are removed.
- Existing full suite behavior should remain compatible except for tests that intentionally codified a now-fixed defect.
- Return a concise summary of changed production behavior and any residual risks.
