Task ID: R46
Short name: review class-group balanced entropy
Relation to overall task: Independent review of W36 SDK hybrid heuristic before benchmarking it.

Assumptions and resolved ambiguities:
- W36 claims it implemented `class_group_balanced_entropy` and full tests pass.
- Reviewer is read-only and must not edit files.
- Worktree has unrelated dirty files; focus on W36 scope.

Goal and expected result:
- Determine whether `class_group_balanced_entropy` correctly combines predicted-class balancing and group-diverse selection, is deterministic, registered, exported, and adequately tested.
- Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvements.

Responsibility boundaries:
- Review scope:
  - `src/active_learning_sdk/strategies/uncertainty.py`
  - `src/active_learning_sdk/strategies/__init__.py`
  - `src/active_learning_sdk/engine.py`
  - `tests/test_class_group_balanced_entropy_strategy.py`
  - interaction with class/group strategy tests if relevant
- Out of scope:
  - `benchmarks/**`
  - README/docs
  - Docker/backends

Review checks:
- Uses shared probability normalization and deterministic tie-breaking patterns.
- Missing `group_id` is treated as isolated per sample.
- Avoids repeated groups before fallback when enough distinct groups exist.
- Maintains class-balanced round-robin behavior.
- Fills deterministically when classes/groups are exhausted.
- Registered in built-ins, fallback lookup, and public exports.
- Tests prove behavior and edge cases.
- No unrelated changes in benchmark files.

Expected validations:
- `uv run --group dev pytest tests/test_class_group_balanced_entropy_strategy.py -q`
- If cheap, run combined strategy tests.

Acceptance criteria:
- If satisfied, explicitly say there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide concrete findings with severity and evidence.

Parallel/sequential notes:
- This review gates benchmark task for the hybrid strategy.
