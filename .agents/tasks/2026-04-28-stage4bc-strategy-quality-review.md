# Review Stage 4B/4C: Strategy Quality Fixes

## Context

Stage 4B fixed BADGE/committee fail-closed contracts. Stage 4C made embedding/diversity strategy behavior more distinct.

## Goal

Review the Stage 4B/4C implementation as a senior strategy reviewer. Confirm it improves correctness without introducing formula, compatibility, determinism, or test-quality regressions.

## Read Scope

- `src/active_learning_sdk/strategies/badge.py`
- `src/active_learning_sdk/strategies/stochastic.py`
- `src/active_learning_sdk/strategies/embedding.py`
- `src/active_learning_sdk/strategies/hybrid.py`
- `tests/test_badge_strategy.py`
- `tests/test_stochastic_committee_strategies.py`
- `tests/test_embedding_strategies.py`
- `tests/test_hybrid_strategy_framework.py`
- `.agents/tmp/2026-04-28-stage4a-strategy-quality-audit.md`

## Write Scope

- `.agents/tmp/2026-04-28-stage4bc-strategy-quality-review.md` only

## Review Questions

- Does BADGE now fail closed only for malformed present `predict_proba`, while still working when probability support is absent?
- Do committee strategies reject one-member committees with actionable errors?
- Are the new embedding/diversity definitions defensible and deterministic?
- Do the new tests actually detect the prior alias/regression risks, or are they brittle implementation snapshots?
- Does hybrid diversity now reflect its advertised components without overclaiming?
- Are there any P1/P2 regressions, public API breaks, or hidden benchmark-claim risks?

## Constraints

- Review only, do not edit production or tests.
- If rejecting a change, provide exact reproduction or exact reasoning.

## Suggested Validation

```powershell
uv run pytest tests\test_badge_strategy.py tests\test_stochastic_committee_strategies.py tests\test_embedding_strategies.py tests\test_hybrid_strategy_framework.py -q
uv run pytest -q
```

## Acceptance Criteria

- Review file exists.
- No P1/P2 findings remain, or findings are concrete enough for a fix worker.
- Explicitly state whether Stage 4B/4C can be accepted.
