# W53 - Stage 1 Sklearn Text Adapter Baseline

## Context
Stage 1 requires a real production adapter baseline. The SDK currently has a Hugging Face scaffold, but no concrete fast training adapter suitable for public smoke tests and benchmarks.

## Goal
Add a scikit-learn text-classification adapter with real `fit`, `predict_proba`, `evaluate`, and stable model-id behavior.

## Responsibility Boundaries
You own the sklearn adapter implementation and its tests.

## In Scope
- New file `src/active_learning_sdk/adapters/sklearn.py`
- `src/active_learning_sdk/adapters/__init__.py`
- New tests in `tests/test_sklearn_adapter.py`

## Out of Scope
- Do not edit strategy capability contracts.
- Do not edit `engine.py` unless absolutely necessary; coordinate if blocked.
- Do not edit README or benchmark docs.
- Do not edit benchmark scripts.
- Do not edit dependency files; scikit-learn is already a core dependency in this repo.

## Required Adapter Behavior
- Provide a public class, suggested name: `SklearnTextClassifierAdapter`.
- Accept either:
  - a ready sklearn estimator/pipeline that supports text input; or
  - no estimator, in which case construct a fast default text pipeline.
- Default should train quickly on tiny datasets, for example:
  - `TfidfVectorizer`
  - `LogisticRegression` with deterministic `random_state`
- Implement:
  - `fit(texts, labels, **kwargs)`
  - `predict_proba(texts, batch_size=32)`
  - `evaluate(texts, labels)`
  - `get_model_id()`
- Handle estimators without native `predict_proba` if feasible via `decision_function` + stable softmax/sigmoid conversion; otherwise raise `ModelAdapterError` with clear message.
- Preserve label order after fitting:
  - expose probabilities in the fitted classifier's class order;
  - tests should verify row lengths and sums.
- `evaluate()` should return at least:
  - `accuracy`
  - `macro_f1`
  - `weighted_f1`
- Empty inputs should return sensible empty outputs for prediction/evaluation or clear errors where sklearn cannot fit.
- Fit should reject mismatched text/label lengths clearly.
- `get_model_id()` should change after successful fit so prediction caches do not reuse stale model outputs. Use a monotonic internal version counter and estimator class/config fingerprint; do not hash training data.

## Test Requirements
Add tests covering:
- Default adapter fits a tiny text dataset and predicts probability rows that sum to 1.
- `evaluate()` returns accuracy/macro_f1/weighted_f1.
- `get_model_id()` changes after `fit()`.
- Mismatched fit lengths raise `ModelAdapterError`.
- Adapter works with an injected sklearn pipeline/estimator.
- Fallback or clear error behavior for estimator without `predict_proba`.

## Files That Must Not Be Touched
- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/strategies/**`
- `src/active_learning_sdk/engine.py`
- `README.md`
- `benchmarks/**`
- `pyproject.toml`
- `uv.lock`

## Important Constraints
- Keep defaults deterministic and fast.
- Avoid adding dependencies.
- Do not make adapter tests depend on internet or downloaded datasets.
- Do not use notebooks.

## Forbidden Actions
- Do not run destructive git commands.
- Do not revert unrelated dirty worktree changes.

## Execution Plan
1. Implement adapter.
2. Export it from `active_learning_sdk.adapters`.
3. Add focused tests.
4. Run adapter tests and then full pytest.

## Acceptance Criteria
- Full tests pass.
- A public sklearn adapter exists and is usable in a full active-learning smoke project.
