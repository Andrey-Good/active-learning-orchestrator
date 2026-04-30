# R84 - Review Stage 5 Hybrid Framework Core

## Context
W66 implemented `SchedulerConfig(mode="hybrid")` and the hybrid strategy framework.

## Goal
Review hybrid core for config validation, capability fail-fast behavior, score normalization safety, deterministic selection, and guardrail correctness.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W66-owned files.

## In Scope
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/__init__.py`
- `src/active_learning_sdk/strategies/hybrid.py`
- `tests/test_hybrid_strategy_framework.py`
- `tests/test_strategy_capabilities.py` if touched/impacted

## Out of Scope
- Do not edit files.
- Do not review benchmark wiring; not implemented yet.
- Do not implement Stage 6.

## Review Questions
- Is `SchedulerConfig(mode="hybrid", hybrid=...)` validated correctly while preserving backward compatibility?
- Does hybrid capability validation require `predict_proba` and `embed` as appropriate?
- Are unknown components and invalid weights/config values rejected clearly?
- Does score normalization handle constant, extreme finite, NaN/inf cases safely?
- Are weighted and prefilter modes deterministic and duplicate-free?
- Do class/group guardrails work without breaking missing-group cases?
- Do snapshots include enough config/fallback details?
- Are tests strong enough?

## Validation
- `uv run --group dev pytest -q tests/test_hybrid_strategy_framework.py tests/test_strategy_capabilities.py tests/test_mix_interleaved_scheduler.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No blocking hybrid core findings remain.
