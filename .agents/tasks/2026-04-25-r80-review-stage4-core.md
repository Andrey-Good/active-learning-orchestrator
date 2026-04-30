# R80 - Review Stage 4 Stochastic And Committee Core

## Context
W63 implemented stochastic uncertainty and committee disagreement strategies.

## Goal
Review Stage 4 core for mathematical correctness, validation rigor, capability integration, and scheduler registration.

## Responsibility Boundaries
- This is a read-only review.
- Focus on W63-owned files.

## In Scope
- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/__init__.py`
- `src/active_learning_sdk/strategies/stochastic.py`
- `tests/test_stochastic_committee_strategies.py`
- `tests/test_strategy_capabilities.py`

## Out of Scope
- Do not edit files.
- Do not review benchmark wiring; not implemented yet.
- Do not implement adapters or training.

## Review Questions
- Does `ModelCapabilities` correctly include `predict_committee` and reject protocol stubs?
- Do `SelectionContext.predict_stochastic` and `predict_committee` wrap missing/adaptor errors correctly?
- Are all eight strategy names registered/exported?
- Do stochastic strategies require `predict_stochastic`?
- Do committee strategies require `predict_committee`?
- Is 3D probability validation strict and safe?
- Are formulas for entropy, BALD, variation ratio, variance, vote entropy, KL divergence, pairwise disagreement, and committee margin coherent?
- Are edge cases and scheduler behavior tested?

## Validation
- `uv run --group dev pytest -q tests/test_stochastic_committee_strategies.py tests/test_strategy_capabilities.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not edit files.
- Do not run destructive git commands.

## Acceptance Criteria
- No blocking Stage 4 core findings remain.
