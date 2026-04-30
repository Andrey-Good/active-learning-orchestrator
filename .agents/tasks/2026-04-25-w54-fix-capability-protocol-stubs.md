# W54 - Fix Capability Inspector Protocol Stub Detection

## Context
R68 found a P1 bug: `inspect_model_capabilities()` treats inherited `TextClassificationAdapter` protocol stubs as real capabilities when a concrete class subclasses the protocol but does not override optional methods.

## Goal
Make capability inspection reject protocol-inherited stub methods unless the concrete adapter overrides them.

## Responsibility Boundaries
Own only the capability inspector fix and focused tests.

## In Scope
- `src/active_learning_sdk/adapters/base.py`
- `tests/test_strategy_capabilities.py`

## Out of Scope
- Do not edit engine validation unless the fix cannot be done in the inspector.
- Do not edit strategy files unless absolutely necessary.
- Do not edit sklearn adapter files.
- Do not edit docs or benchmarks.

## Required Fix
- Detect when a method resolved on an adapter is the inherited stub from `TextClassificationAdapter`.
- Treat inherited stubs as unsupported with a clear reason such as `not implemented on adapter`.
- Preserve behavior for real concrete methods.
- Preserve behavior for methods decorated with `unsupported_adapter_method`.
- Ensure this works for optional methods (`embed`, `predict_logits`, `gradient_embed`, `predict_stochastic`, `save`, `load`) and required methods (`predict_proba`, `fit`, `evaluate`, `get_model_id`).

## Tests
Add or update tests so:
- A class subclassing `TextClassificationAdapter` but implementing only `fit`/`evaluate` does not get `predict_proba=True` or `embed=True`.
- The unsupported reason appears in missing capability errors for an entropy scheduler.
- Existing capability tests still pass.

## Validation
- `uv run --group dev pytest -q tests/test_strategy_capabilities.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not revert unrelated changes.

## Acceptance Criteria
- R68 P1 is fixed.
- Full tests pass.
