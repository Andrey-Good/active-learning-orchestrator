# Task W98-A: Runtime State And JSONL Cache Integrity

## Context
Stress review added strict xfail probes in `tests/test_acceptance_runtime_state_2026_04_27.py`.

## Goal
Turn the runtime/cache xfails green by fixing production behavior, not weakening tests.

## Ownership
May change:
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/cache.py`
- `tests/test_acceptance_runtime_state_2026_04_27.py`
- narrowly related existing tests if expectations must be updated to match stricter contracts

Must not change:
- strategy implementations
- public split/prelabel behavior
- benchmark harnesses

## Problems To Fix
1. `project.validate()` misses samples with `sample_status == "labeled"` but no persisted label.
2. `JsonlDiskCacheStore.get()` trusts a corrupt stale index entry and can return another key's value.

## Expected Behavior
- `validate()` must enforce bidirectional sample label/status consistency:
  - labeled status requires a stored label;
  - stored labels require labeled status;
  - stored labels must be valid under the current label schema.
- JSONL disk cache `get(key)` must verify the record's `"key"` matches the requested key before returning.
- On index mismatch, treat as a miss and repair/delete the stale index entry if feasible.

## Acceptance Criteria
- Focused acceptance runtime tests pass without xfail.
- Existing runtime/cache tests pass.
- No weakening/removal of the audit assertions.
