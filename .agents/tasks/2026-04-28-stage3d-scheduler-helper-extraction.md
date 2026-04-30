# Stage 3D: Scheduler Helper Extraction

## Context

`StrategyScheduler.select_batch` and `_select_mix_interleaved` are D-complexity hotspots. After characterization tests, reduce complexity by extracting helper logic without changing scheduler behavior.

## Dependencies

Run after Stage 3A. Avoid overlap with Stage 3B/3C if those touch `engine.py`.

## Goal

Extract scheduler validation/normalization/mix helper functions or a small internal class, preserving behavior.

## Ownership

You may edit:

- `src/active_learning_sdk/engine.py`
- new internal scheduler module if useful
- focused tests only if needed

Do not edit strategy implementations, runtime state machine, benchmarks, or docs.

## In Scope

- Extract duplicate/out-of-pool validation helpers.
- Extract single/mix/interleaved dispatch helpers if low-risk.
- Preserve scheduler snapshots and fallback counters exactly.

## Constraints

- No public API changes.
- No behavior changes.
- Keep state dict format stable.

## Suggested Validation

- `uv run pytest tests/test_core_refactor_characterization.py tests/test_mix_interleaved_scheduler.py tests/test_strategy_capabilities.py tests/test_hybrid_strategy_framework.py -q`
- `uv run pytest -q`
- `uv run --with radon radon cc src\active_learning_sdk\engine.py -s`

## Acceptance Criteria

- Full suite remains green.
- Scheduler complexity decreases or code paths are visibly easier to review.
