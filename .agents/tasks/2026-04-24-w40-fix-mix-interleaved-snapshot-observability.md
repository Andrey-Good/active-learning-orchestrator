Task ID: W40
Short name: fix mix interleaved snapshot observability
Relation to overall task: Resolve R50 review finding before benchmarking `mix_interleaved`, so scheduler decisions are auditable as a product-quality SDK feature.

Assumptions and resolved ambiguities:
- R50 found `mix_interleaved` is functionally accepted but its snapshot omits group-aware diagnostics.
- Existing `mode="mix"` behavior must remain unchanged.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Extend `mix_interleaved` scheduler snapshots with enough metadata to audit group-aware behavior and fallback.
- Add/adjust focused tests to prove the snapshot reports group lookup success/degradation and group-relaxed fallback information.

Responsibility boundaries:
- In scope:
  - `src/active_learning_sdk/engine.py`
  - `tests/test_mix_interleaved_scheduler.py`
- Out of scope:
  - `benchmarks/**`
  - `src/active_learning_sdk/configs.py` unless strictly necessary
  - strategy implementations
  - docs/README

Architectural constraints and forbidden actions:
- Do not change selection semantics unless absolutely necessary for observability.
- Keep snapshot JSON-serializable.
- Do not include large raw sample payloads.
- Do not change existing `mode="mix"` path.
- Preserve deterministic behavior.

Suggested snapshot fields:
- `group_lookup_available`: boolean, false if group lookup degraded to per-sample ids.
- `selected_group_count`: number of distinct group keys in selected batch.
- `group_constrained_selected_count`: count selected during the first group-constrained pass.
- `group_relaxed_fallback_count`: count selected after dropping group constraints.
- Optionally `selected_group_keys` if compact and JSON-safe; avoid if it makes snapshots too noisy.

High-level execution plan:
- Inspect current `_select_mix_interleaved` and tests.
- Modify group-key helper if needed to return availability metadata.
- Add snapshot fields.
- Update focused tests for normal group-aware path and group-provider-failure degradation.
- Run focused and full tests.

Step -> verify:
- Snapshot fields added -> verify focused tests assert them.
- Degraded group lookup -> verify snapshot says unavailable.
- Existing behavior unchanged -> verify old tests still pass.

Acceptance criteria:
- R50 finding is resolved.
- `uv run --group dev pytest tests/test_mix_interleaved_scheduler.py -q` passes.
- `uv run --group dev pytest -q` passes.

Expected tests and validations:
- Focused scheduler tests.
- Full suite if feasible.

Dependencies:
- Fixes R50.

Parallel/sequential notes:
- Required before benchmark use of `mix_interleaved`.
