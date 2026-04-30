Task ID: R42
Short name: review class-balanced entropy strategy
Relation to overall task: Independent review of W32 SDK heuristic implementation before it is used in benchmark experiments.

Assumptions and resolved ambiguities:
- W32 claims it implemented and registered `class_balanced_entropy`.
- Reviewer is read-only and must not edit files.
- The worktree contains many unrelated pre-existing dirty changes; focus only on the W32 scope unless an interaction creates a real defect.

Goal and expected result:
- Determine whether `class_balanced_entropy` is correct, deterministic, well-tested, registered consistently, and architecturally clean.
- Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvements.

Responsibility boundaries:
- Review scope:
  - `src/active_learning_sdk/strategies/uncertainty.py`
  - `src/active_learning_sdk/strategies/__init__.py`
  - `src/active_learning_sdk/engine.py`
  - `tests/test_class_balanced_entropy_strategy.py`
  - interactions with existing strategy tests if relevant
- Out of scope:
  - benchmark artifacts
  - docs/README
  - Docker/backends unrelated to strategy registration

Review checks:
- Does the strategy use existing probability normalization and deterministic tie-breaking patterns?
- Does it balance across predicted classes without losing top-entropy ordering inside each class?
- Does it handle `k <= 0`, empty pool, `k > pool`, one predicted class, malformed probabilities via shared validation?
- Is it registered in built-ins and exports so users can select it via config?
- Are tests adequate and meaningful rather than only checking happy paths?
- Does the diff avoid unrelated changes?

Expected validations:
- Run `uv run --group dev pytest tests/test_class_balanced_entropy_strategy.py -q`.
- If cheap, run related strategy tests.

Acceptance criteria:
- If satisfied, explicitly say there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide concrete findings with severity and evidence.

Parallel/sequential notes:
- This is a review of W32 and can run while W33 benchmark probe continues.
