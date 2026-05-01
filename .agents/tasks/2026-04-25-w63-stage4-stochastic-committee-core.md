# W63 - Stage 4 Stochastic And Committee Strategy Core

## Context
Stage 4 adds uncertainty methods beyond a single deterministic probability pass. Stage 1 already added `predict_stochastic` capability detection, but the SDK does not yet expose stochastic/committee strategies.

## Goal
Implement stochastic uncertainty and committee disagreement strategies with strict shape validation and fail-fast capability checks.

## Responsibility Boundaries
Own the Stage 4 core strategy slice.

## In Scope
- `src/active_learning_sdk/adapters/base.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/__init__.py`
- New file `src/active_learning_sdk/strategies/stochastic.py`
- New tests in `tests/test_stochastic_committee_strategies.py`
- `tests/test_strategy_capabilities.py` if needed

## Out of Scope
- Do not edit benchmarks yet.
- Do not edit sklearn adapter.
- Do not edit docs/README.
- Do not edit dependency files.
- Do not implement real ensemble training or MC-dropout internals; adapters provide predictions.

## Required Capabilities
- Keep `predict_stochastic(texts, n=..., batch_size=...)`.
- Add committee capability if needed:
  - suggested method: `predict_committee(texts, batch_size=...)`.
- Extend `ModelCapabilities`, `TextClassificationAdapter`, and `inspect_model_capabilities()` for committee support.
- Protocol-inherited stubs must still be rejected as `not implemented on adapter`.

## Required SelectionContext Hooks
- `SelectionContext.predict_stochastic(sample_ids, n=10, batch_size=32)`.
- `SelectionContext.predict_committee(sample_ids, batch_size=32)` if a separate committee capability is added.
- Missing capability should raise `ConfigurationError`.
- Adapter failures should be wrapped as `ModelAdapterError`.

## Required Strategies
Implement built-in strategy names:
- `mc_dropout_entropy`
- `bald`
- `variation_ratio`
- `prediction_variance`
- `committee_vote_entropy`
- `committee_kl_divergence`
- `committee_pairwise_disagreement`
- `committee_margin`

## Required Behavior
- Stochastic strategies require `predict_stochastic`.
- Committee strategies require `predict_committee` if that capability is added.
- Validate stochastic/committee output as 3D probabilities:
  - rows equal requested sample count;
  - each sample has at least one pass/member;
  - each pass/member row is non-empty, numeric, finite, non-negative;
  - row sums must be positive and should be normalized if not exactly 1;
  - class width must be consistent across all rows.
- Strategy scores:
  - `mc_dropout_entropy`: entropy of mean probability.
  - `bald`: entropy(mean probability) - mean(entropy(each pass)).
  - `variation_ratio`: `1 - most_common_predicted_class_count / n_passes`.
  - `prediction_variance`: mean variance across class probabilities over passes.
  - `committee_vote_entropy`: entropy of committee argmax vote distribution.
  - `committee_kl_divergence`: mean KL divergence from members to consensus probability.
  - `committee_pairwise_disagreement`: mean pairwise class-vote disagreement.
  - `committee_margin`: uncertainty from smallest top-two vote margin, e.g. `1 - margin`.
- Deterministic tie-breaking aligned with existing strategy style.
- Handle empty pool, `k <= 0`, `k > pool`, duplicate pool ids.

## Registration
- Add strategies to `_built_in_strategies()`, scheduler lookup, and public strategy exports.
- Configure/attach should fail fast for missing capabilities before `run_step()`.

## Test Requirements
Add tests covering:
- Each stochastic strategy selects expected/highest-scoring sample on small hand-built stochastic predictions.
- Each committee strategy selects expected/highest-scoring sample on small committee predictions.
- Shape validation failures raise `ConfigurationError`.
- Missing stochastic/committee capability fails at configure.
- Models with protocol-inherited stubs do not falsely pass capability detection.
- Scheduler can select with representative stochastic and committee strategy names.

## Validation
- `uv run --group dev pytest -q tests/test_stochastic_committee_strategies.py tests/test_strategy_capabilities.py`
- `uv run --group dev pytest -q`

## Forbidden Actions
- Do not run destructive git commands.
- Do not modify benchmark result artifacts.
- Do not revert unrelated changes.

## Acceptance Criteria
- All required Stage 4 core strategies are built in.
- Full tests pass.
- Stage 4 benchmark wiring can be done as a separate subtask.
