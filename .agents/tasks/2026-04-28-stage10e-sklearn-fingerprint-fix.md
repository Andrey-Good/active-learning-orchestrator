# Stage 10E: Sklearn Fitted-State Fingerprint Fix

## Task Identifier

stage10e-sklearn-fingerprint-fix

## Context

Stage 10D review rejected the adapter/capability release because the sklearn
adapter's model id can still collide for common fitted estimators. The existing
fix fingerprints a small allowlist of learned attributes and misses
`MultinomialNB` state.

## Goal

Close the remaining P1 cache-poisoning blocker by making sklearn fitted-state
fingerprinting broad enough for common sklearn estimators and adding a regression
test.

## Responsibility Boundaries

In scope:

- `src/active_learning_sdk/adapters/sklearn.py`
- `tests/test_sklearn_adapter.py`

Out of scope:

- Engine custom strategy changes.
- Hugging Face adapter changes.
- Docs unless required by the exact fix.
- Benchmark artifacts.

## Files May Be Changed

- `src/active_learning_sdk/adapters/sklearn.py`
- `tests/test_sklearn_adapter.py`

## Files Must Not Be Touched

- Other production modules.
- Public docs.
- Benchmark files.

## Architectural Constraints

- `get_model_id()` must remain deterministic.
- It must distinguish same-config, differently fitted common sklearn estimators.
- Avoid relying on object identity or process-local memory addresses.
- Avoid serializing obviously huge/private/callable/cache-only internals.
- If a value cannot be safely serialized, represent it by stable class/type/shape
  information rather than failing `get_model_id()`.

## Required Fix

- Broaden `_fitted_state()` so it includes learned public fitted attributes ending
  with `_` in addition to any curated attributes.
- Include common Naive Bayes attributes such as `class_log_prior_`,
  `feature_log_prob_`, `class_count_`, and `feature_count_`.
- Preserve nested pipeline step traversal.
- Add a regression test using same-config `CountVectorizer` + `MultinomialNB`
  pipelines with inverted labels; assert `get_model_id()` differs and
  probabilities differ.

## Expected Tests

Run:

- `uv run pytest tests/test_sklearn_adapter.py -q`
- `uv run pytest tests/test_strategy_capabilities.py tests/test_sklearn_adapter.py tests/test_huggingface_adapter.py -q`

If fast enough, also run:

- `uv run pytest -q`

## Acceptance Criteria

- The Stage 10D reviewer reproduction no longer collides.
- Existing sklearn tests still pass.
- No unrelated edits.

## Dependencies

- Stage 10D review report.

## Parallelism

Single narrow worker task. Requires a separate review after implementation.
