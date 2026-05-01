# Stage 4D Fix: Non-Single Strategy Diagnostics

## Context

Stage 4D review rejected the diagnostics implementation because only `single` mode clears/consumes strategy diagnostics. Fallback-capable strategies can also run through `mix`, `mix_interleaved`, and `bandit`, so diagnostics can be omitted from snapshots and left on a reused context.

## Goal

Make strategy diagnostics scoped and consumed wherever built-in strategies are invoked by scheduler modes.

## Ownership

May edit:

- `src/active_learning_sdk/engine.py`
- tests that cover scheduler snapshots/diagnostics, preferably `tests/test_core_refactor_characterization.py` or existing scheduler tests

Must not edit:

- strategy implementation files
- benchmark scripts
- docs
- backend/state/cache files

## In Scope

- Ensure diagnostics are cleared before invoking a strategy arm in single, mix, mix_interleaved, and bandit paths.
- Ensure diagnostics are consumed after the strategy arm returns, so they cannot leak into later calls.
- Surface non-single diagnostics in snapshots under a JSON-safe structured key such as `strategy_diagnostics`, without overwriting mode-specific fields.
- Keep existing non-fallback snapshots unchanged where no diagnostics occurred.
- Add tests for at least mix and bandit or mix_interleaved to prove diagnostics are surfaced and do not leak.

## Out Of Scope

- Do not redesign the public strategy protocol.
- Do not change strategy return types.
- Do not modify strategy math.

## Acceptance Criteria

- Single fallback snapshot behavior remains accepted.
- Mix/bandit or mix_interleaved fallback diagnostics appear in snapshots and are consumed.
- Non-fallback snapshots remain compatible.
- Focused scheduler/strategy tests and full suite pass.

## Suggested Validation

```powershell
uv run pytest tests\test_core_refactor_characterization.py tests\test_mix_interleaved_scheduler.py tests\test_hybrid_strategy_framework.py -q
uv run pytest -q
uv run mypy src
uv run --with ruff ruff check .
```
