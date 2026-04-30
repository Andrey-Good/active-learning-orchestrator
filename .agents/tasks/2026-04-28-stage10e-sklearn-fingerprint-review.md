# Stage 10E Review: Sklearn Fitted-State Fingerprint Fix

## Task Identifier

stage10e-sklearn-fingerprint-review

## Context

Stage 10D review rejected the release because sklearn `get_model_id()` still
collided for same-config differently fitted `CountVectorizer` + `MultinomialNB`
pipelines. Stage 10E claims to fix this narrowly.

## Goal

Verify the remaining P1 cache-poisoning blocker is actually closed without
introducing new serious problems.

## Responsibility Boundaries

In scope:

- `src/active_learning_sdk/adapters/sklearn.py`
- `tests/test_sklearn_adapter.py`
- Stage 10D review reproduction

Out of scope:

- Editing files.
- Reviewing unrelated engine/HF changes beyond noting obvious conflicts.

## Files May Be Changed

- Only `.agents/tmp/2026-04-28-stage10e-sklearn-fingerprint-review.md`

## Review Questions

1. Does the `MultinomialNB` reproduction now produce different model ids?
2. Does fitted-state fingerprinting include public learned attributes broadly
   enough for common sklearn estimators?
3. Is the fingerprint deterministic across calls for the same fitted estimator?
4. Are there obvious huge/private/callable state hazards?
5. Does the regression test actually fail on the old broken implementation and
   pass now?

## Expected Validation

Run at least:

- `uv run pytest tests/test_sklearn_adapter.py -q`

Optional:

- `uv run pytest tests/test_strategy_capabilities.py tests/test_sklearn_adapter.py tests/test_huggingface_adapter.py -q`

## Acceptance Criteria

- Write accept/reject verdict.
- Any blocker must include exact evidence.
- If accepted, list residual P3 risks only.

## Dependencies

- Stage 10E worker patch.
