Task ID: W36
Short name: class-group balanced entropy strategy
Relation to overall task: Next SDK algorithm improvement based on benchmark evidence that class balancing improves macro-F1 while group balancing controls redundant group concentration.

Assumptions and resolved ambiguities:
- `class_balanced_entropy` is accepted and benchmarked.
- `group_diverse_entropy` is accepted and benchmarked.
- Class balance and group diversity have complementary strengths.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Implement a built-in SDK strategy named `class_group_balanced_entropy`.
- The strategy should balance across predicted classes like `class_balanced_entropy`, while preferring not to repeat `group_id` across the batch when possible.
- It should be deterministic and robust with the same probability validation behavior as existing strategies.

Responsibility boundaries:
- In scope:
  - `src/active_learning_sdk/strategies/uncertainty.py`
  - `src/active_learning_sdk/strategies/__init__.py`
  - `src/active_learning_sdk/engine.py`
  - focused tests under `tests/`
- Out of scope:
  - `benchmarks/**`
  - README/docs
  - Docker/backends
  - broad scheduler refactors

Architectural constraints and forbidden actions:
- Reuse existing helpers/patterns from `ClassBalancedEntropyStrategy` and `GroupDiverseEntropyStrategy`.
- Do not add dependencies.
- Do not change existing strategy behavior.
- Do not access private engine/project state.
- Missing `group_id` should be treated as an isolated per-sample group, matching `group_diverse_entropy`.
- If `k` exceeds the number of unique groups/classes, fill remaining slots deterministically by class-balanced entropy order.

High-level execution plan:
- Inspect current class/group strategies.
- Implement `ClassGroupBalancedEntropyStrategy`.
- Register and export it.
- Add tests for:
  - predicted-class round-robin behavior;
  - no repeated groups when alternatives exist;
  - deterministic fill when groups/classes are exhausted;
  - missing group IDs as per-sample groups;
  - scheduler availability.
- Run focused and full tests.

Step -> verify:
- Strategy implementation -> verify direct tests pass.
- Registration/export -> verify scheduler can select it by config.
- Edge cases -> verify `k <= 0`, empty pool, duplicate ids, missing groups.

Acceptance criteria:
- `class_group_balanced_entropy` is usable through `SchedulerConfig(mode="single", strategy="class_group_balanced_entropy")`.
- It avoids repeated groups before fallback when enough distinct groups exist.
- It maintains class-balanced ordering across predicted classes.
- Tests cover edge cases and deterministic behavior.

Expected tests and validations:
- `uv run --group dev pytest tests/test_class_group_balanced_entropy_strategy.py -q`
- `uv run --group dev pytest tests/test_class_balanced_entropy_strategy.py tests/test_group_diverse_strategy.py tests/test_class_group_balanced_entropy_strategy.py -q`
- `uv run --group dev pytest -q` if feasible.

Dependencies:
- Sequentially after R45.

Parallel/sequential notes:
- This is SDK-source work and should be reviewed independently before benchmark use.
