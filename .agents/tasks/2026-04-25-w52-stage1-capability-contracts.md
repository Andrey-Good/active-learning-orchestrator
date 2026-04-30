# W52 - Stage 1 Strategy Capability Contracts

## Context
Stage 1 goal is to stop strategies from relying on implicit model methods. The current engine checks only global MVP methods (`predict_proba`, `fit`, `evaluate`) and has a special-case rejection for `coreset_kcenter`.

## Goal
Implement first-class strategy capability declarations and scheduler validation so configured strategies fail fast with explicit missing-capability errors.

## Responsibility Boundaries
You own the strategy capability contract and engine validation path.

## In Scope
- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/strategies/base.py`
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/engine.py`
- New tests in `tests/test_strategy_capabilities.py`

## Out of Scope
- Do not implement sklearn adapter.
- Do not edit `src/active_learning_sdk/adapters/__init__.py`.
- Do not edit README or benchmark docs.
- Do not implement CoreSet/BADGE/stochastic strategies.
- Do not edit benchmark scripts.

## Required Behavior
- Extend `ModelCapabilities` and `inspect_model_capabilities()` to include at least:
  - `predict_proba`
  - `predict_logits`
  - `embed`
  - `gradient_embed`
  - `predict_stochastic`
  - `fit`
  - `evaluate`
  - `get_model_id`
  - `save_load`
- Add a stable strategy capability declaration mechanism. Prefer a simple attribute such as:
  - `required_capabilities: frozenset[str] = frozenset({...})`
- Every built-in strategy must declare requirements:
  - `random`: no model inference capability required.
  - `entropy`, `margin`, `least_confidence`, `class_balanced_entropy`, `group_diverse_entropy`, `class_group_balanced_entropy`: `predict_proba`.
  - `coreset_kcenter`: `embed` and still unsupported/not implemented in this build.
- `configure()` and `attach_runtime()` must validate capabilities for all configured strategy names:
  - `single`
  - `mix`
  - `mix_interleaved`
  - `bandit`
  - custom mode can require only the global train/eval baseline because arbitrary callbacks cannot be introspected yet.
- Missing strategy capabilities must raise `ConfigurationError` before a run starts, with a message naming:
  - strategy name;
  - missing capability;
  - unsupported method reason when available.
- Unknown strategies should still fail clearly.
- Existing explicit `coreset_kcenter` rejection should remain, but route it through the same capability/support validation where practical.
- Keep strict capability semantics compatible with existing tests and benchmark harness.

## Test Requirements
Add tests covering:
- `random` can configure with a model that has `fit`/`evaluate` but no `predict_proba`.
- `entropy` fails during `configure()` when `predict_proba` is missing.
- `mix` or `mix_interleaved` fails if any configured arm lacks required capability.
- `coreset_kcenter` fails during `configure()` with a clear unsupported/not implemented message.
- `inspect_model_capabilities()` recognizes the new optional methods and decorated unsupported placeholders.
- Shape validation tests that already exist must keep passing; do not weaken probability output validation.

## Files That Must Not Be Touched
- `src/active_learning_sdk/adapters/sklearn.py`
- `src/active_learning_sdk/adapters/__init__.py`
- `README.md`
- `benchmarks/**`
- `pyproject.toml`
- `uv.lock`

## Important Constraints
- Do not introduce heavy optional dependencies.
- Keep changes deterministic.
- Preserve custom strategy support.
- Do not degrade current `47 passed` baseline.

## Forbidden Actions
- Do not run destructive git commands.
- Do not revert unrelated dirty worktree changes.
- Do not implement future Stage 2+ strategies.

## Execution Plan
1. Extend capability inspection.
2. Add strategy-level required capability metadata.
3. Replace special-case scheduler validation with generic configured-strategy capability validation plus unsupported strategy rejection.
4. Add focused tests.
5. Run `uv run --group dev pytest -q`.

## Acceptance Criteria
- Full tests pass.
- Strategy-specific missing capabilities fail at `configure()`/`attach_runtime()`.
- Error messages are actionable and name the strategy and capability.
