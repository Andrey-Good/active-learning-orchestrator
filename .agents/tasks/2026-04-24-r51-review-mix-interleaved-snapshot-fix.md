Task ID: R51
Short name: review mix interleaved snapshot fix
Relation to overall task: Verify W40 resolved the R50 observability finding before benchmarking `mix_interleaved`.

Assumptions and resolved ambiguities:
- W40 claims it added group-aware snapshot metadata without changing selection semantics.
- Reviewer is read-only and must not edit files.

Goal and expected result:
- Confirm R50 finding is resolved.
- Confirm tests cover normal group-aware behavior, group-relaxed fallback, and group-provider degradation.
- Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvements.

Responsibility boundaries:
- Review scope:
  - `src/active_learning_sdk/engine.py`
  - `tests/test_mix_interleaved_scheduler.py`
- Out of scope:
  - `benchmarks/**`
  - `src/active_learning_sdk/configs.py` unless needed to understand interaction
  - strategy implementation files

Review checks:
- Snapshot fields are JSON-serializable and accurately named.
- `group_lookup_available` is false when group lookup degrades.
- `group_constrained_selected_count` and `group_relaxed_fallback_count` reflect actual selection phases.
- `selected_group_count` is computed from selected samples.
- Existing `mode="mix"` path is untouched.
- Tests remain meaningful and pass.

Expected validations:
- `uv run --group dev pytest tests/test_mix_interleaved_scheduler.py -q`
- Full suite if cheap.

Acceptance criteria:
- If satisfied, explicitly say there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide concrete findings with severity and evidence.

Parallel/sequential notes:
- This review gates benchmark use of `mix_interleaved`.
