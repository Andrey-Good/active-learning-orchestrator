# W29 - Group-Diverse Uncertainty Strategy

## Relation To Overall Task
The accepted baseline matrix shows uncertainty strategies underperform random on `grouped_duplicates` because batches over-concentrate near-duplicate groups. R38 recommends a group-aware diversity reranker as the first measurable SDK strategy improvement.

## Assumptions And Resolved Ambiguities
- This task implements SDK behavior only; benchmark harness registration/runs can be a follow-up task.
- Use existing `DataSample.group_id` exposed through `SelectionContext.get_samples(...)`.
- The strategy should remain deterministic.
- Missing group IDs should not crash; treat each ungrouped sample as its own group.
- The first implementation can be a built-in strategy with fixed conservative defaults rather than a large configurable framework.

## Goal And Expected Result
Add a built-in group-diverse uncertainty strategy that scores uncertainty but reranks selection to avoid repeatedly choosing the same `group_id` in one batch. It should be registered by `StrategyScheduler` and covered by tests.

## Responsibility Boundaries
Owned by this worker:
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/engine.py` only for built-in registration/name lookup
- `src/active_learning_sdk/strategies/__init__.py` if exports need updating
- `tests/**` focused on strategy correctness

Do not change:
- `benchmarks/**`
- `README.md`
- docs/docker
- unrelated SDK features

## In Scope
- New built-in strategy name, recommended: `group_diverse_entropy`.
- Entropy score first, then greedy group-aware reranking.
- A two-pass selection is acceptable: first pick at most one per group by score, then fill remaining slots by score if batch size exceeds group count.
- Deterministic tie-breaking consistent with existing strategies.
- Tests for:
  - prefers different groups when possible;
  - falls back/fills when groups are fewer than `k`;
  - missing group IDs are isolated;
  - output remains unique and clipped;
  - strategy is available through `StrategyScheduler`.

## Out Of Scope
- Benchmark harness strategy registration and result artifacts.
- Embedding-based diversity / k-center / BADGE.
- Configurable group caps.
- Predicted-class balancing.

## Files Or Modules May Be Changed
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/strategies/__init__.py`
- `tests/**`

## Files Or Areas Must Not Be Touched
- `benchmarks/**`
- `README.md`
- `docs/**`
- `docker/**`

## Important Architectural Constraints And Forbidden Actions
- Do not use true labels.
- Do not make randomness global or nondeterministic.
- Do not break existing strategy names.
- Do not silently degrade to random unless needed to fill after scoring.

## High-Level Execution Plan
- Add a helper that produces entropy scores and group-aware greedy rerank.
- Add `GroupDiverseEntropyStrategy`.
- Register it in scheduler built-ins.
- Add tests with a fake context/provider/model.
- Run targeted and full available tests.

## Step -> Verify Plan
- Implement strategy -> test direct `select`.
- Register scheduler -> test `SchedulerConfig(strategy="group_diverse_entropy")`.
- Run available tests.

## Acceptance Criteria
- Strategy is selectable by `SchedulerConfig(mode="single", strategy="group_diverse_entropy")`.
- Direct tests prove reduced group concentration under controlled probabilities.
- Existing tests still pass.

## Expected Tests And Validations
- `uv run --group dev pytest tests/test_group_diverse_strategy.py -q`
- `uv run --group dev pytest -q`

## Dependencies
Depends on accepted baseline/audit.

## Parallel Or Sequential Notes
Benchmark harness update must wait until this SDK implementation is reviewed.
