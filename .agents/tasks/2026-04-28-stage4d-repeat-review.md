# Repeat Review Stage 4D: Non-Single Diagnostics Fix

## Context

Stage 4D was rejected because diagnostics leaked/failed to surface outside `single` mode. A fix worker updated scheduler diagnostics for non-single modes and added tests.

## Goal

Verify the P2 is closed and no new snapshot compatibility, JSON-safety, or scheduler behavior regressions were introduced.

## Read Scope

- `.agents/tmp/2026-04-28-stage4d-diagnostics-evidence-review.md`
- `src/active_learning_sdk/engine.py`
- `tests/test_core_refactor_characterization.py`
- `tests/test_mix_interleaved_scheduler.py`
- related scheduler tests if needed

## Write Scope

- `.agents/tmp/2026-04-28-stage4d-repeat-review.md` only

## Review Questions

- Are diagnostics cleared before and consumed after each built-in arm in single/mix/mix_interleaved/bandit paths?
- Do snapshots surface diagnostics under structured keys without overwriting normal fields?
- Are diagnostics JSON-safe and scoped to the correct strategy invocation?
- Are non-fallback snapshots still minimal/compatible?
- Any P1/P2 regressions?

## Constraints

- Review only, do not edit.

## Suggested Validation

```powershell
uv run pytest tests\test_core_refactor_characterization.py tests\test_mix_interleaved_scheduler.py tests\test_hybrid_strategy_framework.py -q
uv run pytest -q
uv run mypy src
uv run --with ruff ruff check .
```

## Acceptance Criteria

- Explicit accept/reject.
- If accepted, Stage 4D can be treated as complete.
