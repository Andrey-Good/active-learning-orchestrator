Task ID: R50
Short name: review mix interleaved scheduler
Relation to overall task: Independent review of W39 opt-in scheduler implementation before benchmarking it.

Assumptions and resolved ambiguities:
- W39 claims it added `SchedulerConfig(mode="mix_interleaved", mix=...)` without changing existing `mode="mix"` behavior.
- Reviewer is read-only and must not edit files.
- Worktree has unrelated dirty files; focus only on W39 scope.

Goal and expected result:
- Determine whether `mix_interleaved` is opt-in, deterministic, group-aware, has safe fallback behavior, and is adequately tested.
- Verify backward compatibility for existing `mode="mix"`.
- Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvements.

Responsibility boundaries:
- Review scope:
  - `src/active_learning_sdk/configs.py`
  - `src/active_learning_sdk/engine.py`
  - `tests/test_mix_interleaved_scheduler.py`
  - related scheduler tests if relevant
- Out of scope:
  - `benchmarks/**`
  - strategy implementation files unless needed to understand interaction
  - README/docs
  - Docker/backends

Review checks:
- `SchedulerConfig.validate()` accepts `mix_interleaved` and applies mix validation.
- Existing `mode="mix"` path is unchanged or tests prove behavior compatibility.
- New mode uses deterministic quotas and config/insertion order, not sorted-name bias.
- New mode interleaves arms and avoids repeated groups before fallback.
- Missing group provider degrades safely.
- Snapshot contains enough allocation/group/fallback information.
- Tests cover edge cases and are not overly coupled to implementation internals.

Expected validations:
- `uv run --group dev pytest tests/test_mix_interleaved_scheduler.py -q`
- If cheap, run full suite.

Acceptance criteria:
- If satisfied, explicitly say there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide concrete findings with severity and evidence.

Parallel/sequential notes:
- This review gates benchmark use of `mix_interleaved`.
