# Final Repeat Review Stage 4D: Diagnostic Safety Fix

## Context

The previous Stage 4D repeat review rejected two P2 issues:

- non-built-in `Real` diagnostics could serialize as NaN;
- single-mode diagnostics used `snapshot.update(...)` and could overwrite `mode`/`strategy`.

The orchestrator applied a focused fix.

## Goal

Verify both P2 findings are closed and Stage 4D can be accepted.

## Read Scope

- `.agents/tmp/2026-04-28-stage4d-repeat-review.md`
- `src/active_learning_sdk/engine.py`
- `tests/test_core_refactor_characterization.py`
- `tests/test_badge_strategy.py`
- related scheduler diagnostics tests if needed

## Write Scope

- `.agents/tmp/2026-04-28-stage4d-final-repeat-review.md` only

## Review Questions

- Does diagnostic JSON sanitization convert all non-finite `Real` values to JSON-safe `None`?
- Can diagnostics no longer overwrite `mode`, `strategy`, or other top-level single snapshot fields?
- Does accepted fallback information remain visible in snapshots under `strategy_diagnostics`?
- Do focused/full/static validations pass?

## Constraints

- Review only, do not edit.

## Suggested Validation

```powershell
uv run pytest tests\test_badge_strategy.py tests\test_core_refactor_characterization.py tests\test_mix_interleaved_scheduler.py tests\test_hybrid_strategy_framework.py -q
uv run pytest -q
uv run mypy src
uv run --with ruff ruff check .
```

## Acceptance Criteria

- Explicit accept/reject.
- If accepted, Stage 4D is complete.
