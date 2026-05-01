# R68 - Review Stage 1 Capability Contracts

## Context
Worker W52 implemented strategy capability declarations and configure/attach validation.

## Goal
Review W52 changes for correctness, explicit fail-fast behavior, and architectural fit.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W52-owned files and capability validation behavior.

## In Scope
- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/strategies/base.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/engine.py`
- `tests/test_strategy_capabilities.py`

## Out of Scope
- Do not edit files.
- Do not review sklearn adapter implementation except for integration conflicts with capability contracts.
- Do not implement Stage 2+ methods.

## Review Questions
- Are all required model capability flags represented and detected correctly?
- Do built-in strategies declare accurate `required_capabilities`?
- Does `configure()` and `attach_runtime()` validate all configured strategy names across `single`, `mix`, `mix_interleaved`, and `bandit`?
- Does `random` avoid requiring `predict_proba` while still requiring train/evaluate baseline?
- Are missing capability errors clear, actionable, and tied to strategy names?
- Is `coreset_kcenter` still rejected as unsupported in this build?
- Do tests cover edge cases without weakening existing validation?

## Validation
- Run `uv run --group dev pytest -q tests/test_strategy_capabilities.py`.
- Run full `uv run --group dev pytest -q` if feasible.

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No blocking capability-contract findings.
- Validation passes in the combined current workspace.
