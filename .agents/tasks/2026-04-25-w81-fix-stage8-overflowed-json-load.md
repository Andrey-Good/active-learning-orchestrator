# W81 - Fix Stage 8 Overflowed JSON Number Loading

## Context

R99 found a P1 state-safety issue: `json.loads(..., parse_constant=...)` rejects literal `NaN`/`Infinity`, but accepts oversized JSON numbers like `1e999` as `float('inf')`. That can enter `ProjectState` during load.

## Goal

Reject non-finite numeric values after JSON load/state parsing, including overflowed JSON numeric literals.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/state/store.py`
- `tests/test_state_safety.py`

## Acceptance Criteria

- Loading a state file containing `1e999` or `-1e999` raises `StateCorruptedError`.
- Loaded `ProjectState` is strict-JSON-safe according to the same finite-number rules used before save.
- Targeted and full tests pass.

## Validation

- `uv run --group dev pytest -q tests/test_state_safety.py`
- `uv run --group dev pytest -q`
