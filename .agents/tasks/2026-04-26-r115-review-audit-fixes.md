# r115 - Review Audit Fixes

## Context

After w91 and w92 complete, the audit tests should be green and the SDK should enforce stronger runtime/strategy contracts.

## Goal

Perform a read-only review of the audit fixes.

## In Scope

- Runtime validation changes in `src/active_learning_sdk/engine.py`
- Strategy/scheduler contract changes in `src/active_learning_sdk/engine.py` and `src/active_learning_sdk/strategies/uncertainty.py`
- Audit tests

## Out Of Scope

- No edits.
- No benchmark reruns beyond reading artifacts.

## Review Questions

1. Can invalid backend labels still enter `sample_labels`?
2. Are duplicate/unknown provider IDs rejected clearly?
3. Does cached/no-cache prediction output shape mismatch fail with actionable SDK exceptions?
4. Can out-of-pool or duplicate strategy output leak through scheduler selection?
5. Did the fixes introduce over-broad behavior changes that may break existing strategies?

## Acceptance Criteria

- Findings with file/line references.
- Verdict on whether audit blocker issues are resolved.
