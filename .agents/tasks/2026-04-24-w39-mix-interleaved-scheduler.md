Task ID: W39
Short name: mix interleaved scheduler
Relation to overall task: Improve SDK strategy orchestration after R48 found current block-based `mode="mix"` can bias arms and regress group concentration.

Assumptions and resolved ambiguities:
- Existing `mode="mix"` behavior must remain unchanged for backward compatibility.
- This task should add an opt-in scheduler variant, not silently alter existing behavior.
- Previous benchmark evidence suggests group-aware interleaving may preserve rare/early gains while reducing cross-arm duplicate groups.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Add an opt-in scheduler mode named `mix_interleaved`.
- It should use `SchedulerConfig.mix` weights, allocate deterministic quotas, interleave arms rather than concatenate blocks, and prefer candidates whose `group_id` has not already been selected in the current batch.
- It should fall back deterministically to ID-only fill when group-diverse candidates are exhausted.

Responsibility boundaries:
- In scope:
  - `src/active_learning_sdk/configs.py`
  - `src/active_learning_sdk/engine.py`
  - focused tests under `tests/`
- Out of scope:
  - `benchmarks/**`
  - README/docs
  - strategy implementation files unless absolutely required
  - Docker/backends

Architectural constraints and forbidden actions:
- Do not change existing `mode="mix"` behavior.
- Do not introduce dependencies.
- Keep behavior deterministic.
- Use config insertion order for mix arms where reasonable; avoid sorted-name bias in the new mode.
- Missing `group_id` should be treated as isolated per-sample groups.
- If context/provider cannot provide groups, degrade safely to ID-only behavior rather than failing unexpectedly.
- Keep snapshots informative enough for benchmarks/debugging.

High-level execution plan:
- Update `SchedulerConfig.validate()` to accept `mix_interleaved` and require a positive `mix`.
- Add helper(s) in `StrategyScheduler` to allocate deterministic quotas, get group keys, and run interleaved selection.
- Add tests proving:
  - old `mix` remains block-based/current behavior;
  - `mix_interleaved` alternates arms according to quotas/config order;
  - `mix_interleaved` avoids repeated groups before fallback;
  - fallback fills when unique groups are exhausted;
  - snapshots expose requested/actual allocations and mode.
- Run focused scheduler tests and full suite.

Step -> verify:
- Config mode accepted -> verify construction and validation tests.
- Scheduler implementation -> verify deterministic unit tests.
- Backward compatibility -> verify old mix test proves unchanged block order.
- Full suite -> verify no regressions.

Acceptance criteria:
- Users can configure `SchedulerConfig(mode="mix_interleaved", mix={...})`.
- Existing `mode="mix"` behavior is unchanged.
- New mode is deterministic, group-aware, and has fallback behavior.
- Tests cover key edge cases and snapshots.

Expected tests and validations:
- `uv run --group dev pytest tests/test_mix_interleaved_scheduler.py -q`
- `uv run --group dev pytest -q`

Dependencies:
- Sequentially after R49/R48.

Parallel/sequential notes:
- SDK-source work. Must be independently reviewed before benchmark use.
