Task ID: W32
Short name: class-balanced entropy strategy
Relation to overall task: Improve existing active-learning heuristics based on benchmark evidence and add one scientifically testable strategy that addresses uncertainty collapse into one predicted class/region.

Assumptions and resolved ambiguities:
- Existing group-diverse entropy fixed group concentration but did not beat all baselines on grouped duplicates.
- A separate predicted-class-balanced uncertainty heuristic is a reasonable next hypothesis because entropy/margin can over-query one predicted class while random preserves coverage.
- This worker is not alone in the codebase. Other agents may touch benchmark-only files in parallel. Do not revert or overwrite changes outside this task's scope.

Goal and expected result:
- Implement a built-in SDK strategy named `class_balanced_entropy`.
- It should select high-entropy samples while round-robin balancing across model-predicted classes when possible.
- It should be deterministic for a fixed model/pool and robust to invalid `predict_proba` outputs using existing validation helpers.

Responsibility boundaries:
- In scope:
  - `src/active_learning_sdk/strategies/uncertainty.py`
  - `src/active_learning_sdk/strategies/__init__.py`
  - `src/active_learning_sdk/engine.py`
  - focused tests under `tests/`
- Out of scope:
  - benchmark harness files and benchmark result artifacts
  - README or product docs
  - Docker/Label Studio/backends
  - broad refactors unrelated to this strategy

Architectural constraints and forbidden actions:
- Reuse `_normalize_probability_rows`, `_entropy_scores`, `_select_top_scored` patterns where appropriate.
- Do not introduce dependencies.
- Do not access private engine/project state from the strategy.
- Do not make non-deterministic random choices.
- Do not change existing strategy behavior.

High-level execution plan:
- Inspect existing strategy and scheduler registration patterns.
- Implement `ClassBalancedEntropyStrategy`.
- Register it in public exports and built-in strategies.
- Add unit tests for balancing, deterministic tie handling, edge cases, and availability.
- Run the narrow relevant tests, then the full test suite if feasible.

Step -> verify:
- Add strategy class -> verify direct strategy tests pass with fake contexts.
- Register strategy -> verify `ActiveLearningProject.available_strategies()` or scheduler availability sees it.
- Edge cases -> verify `k <= 0`, empty pool, `k > pool`, one predicted class, unnormalized probabilities.

Acceptance criteria:
- `class_balanced_entropy` is usable through `SchedulerConfig(mode="single", strategy="class_balanced_entropy")`.
- It prioritizes uncertainty within each predicted class while preventing one predicted class from monopolizing the batch when multiple predicted classes are available.
- It preserves deterministic outputs.
- Tests prove behavior and edge cases.

Expected tests and validations:
- `uv run --group dev pytest tests/test_group_diverse_strategy.py tests/test_import_labels.py -q` if relevant.
- Add/run the new focused strategy tests.
- Run `uv run --group dev pytest -q` if runtime permits.

Dependencies:
- Depends on W29/W30 group-diverse registration patterns being present.
- Does not depend on W33 benchmark mix probe.

Parallel/sequential notes:
- Can run in parallel with W33 because W33 must not edit SDK strategy/source files.
