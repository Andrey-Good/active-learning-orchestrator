# Stage 4B: Strategy Contract Fixes

## Context

Stage 4A found two concrete strategy contract defects:

- BADGE cold-start probability validation fails open when `predict_proba` exists but returns malformed output.
- Committee strategies accept one-member committees, which makes disagreement formulas degenerate.

These are correctness/validation fixes and should be done before broader strategy-quality changes.

## Goal

Make BADGE and committee strategies fail closed on malformed or degenerate model outputs, with focused regression tests.

## Ownership

May edit:

- `src/active_learning_sdk/strategies/badge.py`
- `src/active_learning_sdk/strategies/stochastic.py`
- `tests/test_badge_strategy.py`
- `tests/test_stochastic_committee_strategies.py`

Must not edit:

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/embedding.py`
- `src/active_learning_sdk/strategies/hybrid.py`
- benchmark scripts
- docs, except if a test needs an inline comment adjustment

## In Scope

- BADGE should skip the optional cold-start probability check only when `predict_proba` is genuinely absent/not callable.
- BADGE should raise `ConfigurationError` when `predict_proba` exists and returns malformed rows.
- Committee probability-cube validation should require at least two members per sample.
- Error messages should mention the method/strategy enough to be actionable.

## Out Of Scope

- Do not redesign BADGE.
- Do not add scheduler snapshots.
- Do not change stochastic MC-dropout pass-count behavior.
- Do not touch external benchmark evidence.

## Architectural Constraints

- Keep deterministic selection behavior unchanged for valid inputs.
- Preserve public strategy names and required capabilities.
- Do not catch broad exceptions in ways that hide malformed model outputs.

## Acceptance Criteria

- New/updated tests prove BADGE fails closed on malformed probability output.
- New/updated tests prove all committee strategies reject one-member committees.
- Existing BADGE and stochastic/committee tests pass.
- Full suite should remain green.

## Suggested Validation

```powershell
uv run pytest tests\test_badge_strategy.py tests\test_stochastic_committee_strategies.py -q
uv run pytest -q
```

## Dependencies

Starts after Stage 4A audit.
