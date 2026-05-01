# Stage 10D: Adapter And Capability Contract Fixes

## Task Identifier

stage10d-adapter-capability-fixes

## Context

Stage 10A/10B/10C read-only audits rejected adapter/capability readiness. This
task fixes only the confirmed P1/P2 blockers without broad architecture churn.

## Goal

Make the adapter/capability surface safe enough for Stage 10 review:

- strict custom strategies can be supplied before capability validation;
- adapter docs consistently distinguish minimal engine adapter methods from
  strategy-specific probability capabilities;
- sklearn adapter model ids are cache-safe for common fitted estimator state;
- sklearn adapter rejects degenerate one-class probability surfaces;
- direct sklearn submodule import has actionable optional-extra guidance;
- Hugging Face `predict_proba()` validates rows and moves tokenizer tensors to
  the model device.

## Responsibility Boundaries

In scope:

- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/adapters/sklearn.py`
- `src/active_learning_sdk/adapters/huggingface.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- focused tests for the above
- small README/contract wording fixes if needed

Out of scope:

- Full Hugging Face training implementation.
- New benchmark datasets.
- Strategy algorithm changes.
- Label backend changes.

## Files May Be Changed

- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/adapters/sklearn.py`
- `src/active_learning_sdk/adapters/huggingface.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- `tests/test_sklearn_adapter.py`
- `tests/test_strategy_capabilities.py`
- new focused HF adapter test file if useful
- `README.md`
- `docs/SDK_CONTRACTS.md`

## Files Must Not Be Touched

- Benchmark artifacts and benchmark result claims.
- Label backend implementation.
- Dataset provider implementation.
- Existing audit reports except this task's worker notes if needed.

## Architectural Constraints

- Root import must stay dependency-light.
- Optional adapter classes may remain lazy-loaded.
- Public API additions must be backward-compatible.
- Do not silently normalize invalid probabilities.
- Do not invent a fake HF training loop.
- Capability strict mode must fail before selection when custom strategies are
  provided at configure/attach time.

## Required Fixes

1. Add a public path to supply custom strategies at configure/attach time.
   Suggested shape: optional `strategies: Sequence[SamplingStrategy] | None = None`
   on `ActiveLearningProject.configure()`, `ActiveLearningProject.attach_runtime()`,
   and the corresponding engine methods. Use those strategies in strict capability
   validation and scheduler registration.
2. Keep post-config `register_strategy()` behavior, but validate registered
   strategy capabilities immediately against the attached model when strict
   capability mode is enabled.
3. Align adapter contract docs: minimal engine adapter requires `fit` and
   `evaluate`; `predict_proba` is required by probability strategies and
   prelabeling, not by random-only orchestration.
4. Harden sklearn `get_model_id()` so same-config, differently fitted estimators
   do not collide in common sklearn state. Include fitted state such as
   `classes_`, `coef_`, `intercept_`, vocabulary/idf, and nested pipeline fitted
   step state where available. Keep output deterministic.
5. Reject one-class fitted estimators through `_classes()` and/or post-fit
   validation with `ModelAdapterError`.
6. Make direct `active_learning_sdk.adapters.sklearn` import fail with a clear
   `active-learning-sdk[sklearn]` message if sklearn is missing.
7. Harden HF `predict_proba()` with finite/non-negative/normalized/min-width
   validation and tokenizer output device movement.

## Tests And Validations

Add focused tests for:

- strict custom strategy configure path succeeds/fails based on model capability;
- post-config `register_strategy()` validates when the scheduler uses that
  strategy and strict mode is enabled;
- sklearn model-id collision regression;
- sklearn one-class custom estimator rejection;
- direct sklearn submodule missing-extra guidance;
- HF invalid logits/probabilities rejection;
- HF tokenizer output movement to model device with lightweight fakes;
- HF scaffold still reports `fit=False` and `evaluate=False`.

Run at least:

- `uv run pytest tests/test_strategy_capabilities.py tests/test_sklearn_adapter.py -q`
- focused HF adapter tests
- `uv run pytest -q`
- `uv run mypy src`
- `uv run --with ruff ruff check .`

## Acceptance Criteria

- Stage 10A/10B/10C P1/P2 blockers are closed.
- No broad unrelated refactors.
- Full suite and static checks pass.
- Any remaining P3 is documented as non-blocking.

## Dependencies

- Stage 10A/10B/10C audits.

## Parallelism

This implementation should be handled by one worker because engine, adapter, docs,
and tests are coupled. Review must be done by a separate reviewer afterward.
