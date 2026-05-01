# Review Stage 3D: Scheduler Helper Extraction

## Context

Stage 3D extracted helper methods from `StrategyScheduler.select_batch` and `_select_mix_interleaved` while preserving scheduler behavior.

## Goal

Review only. Confirm scheduler behavior/state snapshots/fallback counters are preserved and extraction did not regress custom/single/mix/hybrid/bandit modes.

## In Scope

- `src/active_learning_sdk/engine.py`, specifically `StrategyScheduler`
- `tests/test_core_refactor_characterization.py`
- scheduler-related tests

## Review Questions

- Are selected IDs still validated against pool and duplicate-free?
- Are custom selectors still supported with existing call-style behavior?
- Are mix/interleaved snapshots and fallback counters preserved?
- Are hybrid/bandit mode paths still intact?
- Did helper extraction avoid public API/state format changes?

## Constraints

- Review only, do not edit.
- Provide concrete P1/P2 findings if any.

## Suggested Validation

- `uv run pytest tests/test_core_refactor_characterization.py tests/test_mix_interleaved_scheduler.py tests/test_strategy_capabilities.py tests/test_hybrid_strategy_framework.py -q`
- `uv run pytest -q`

## Acceptance Criteria

- No P1/P2 findings.
- Stage 3D can be treated as complete.
