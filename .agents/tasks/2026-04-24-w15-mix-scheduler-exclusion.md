# W15 - Mix Scheduler Exclusion-Aware Allocation

## Relation to Overall Task
SDK behavior improvement for existing `SchedulerConfig(mode="mix")`. Prior audits found mix mode selects each strategy from the full pool and dedups later, which can underfill or distort requested weights.

## Assumptions and Resolved Ambiguities
- This is a correctness/quality-of-scheduler task, not a new heuristic.
- Do not touch benchmark runner.
- Do not change strategy scoring logic.

## Goal
Make mix scheduler allocation exclusion-aware:
- each component strategy should see only remaining unselected pool items;
- fallback should also select only from remaining pool;
- final selection should fill up to `k` when enough pool items exist;
- snapshot should include requested and actual allocations for diagnostics.

## Responsibility Boundaries
- Own only scheduler code and focused tests.

## In Scope
- `src/active_learning_sdk/engine.py`
- focused tests under `tests/`
- Add tests with overlapping deterministic strategies to prove allocation/fill behavior.
- Preserve existing public config shape.

## Out of Scope
- Do not edit `src/active_learning_sdk/strategies/uncertainty.py`.
- Do not edit benchmark runner/artifacts.
- Do not implement bandit.
- Do not add new scheduler modes.

## Files/Modules May Change
- `src/active_learning_sdk/engine.py`
- `tests/test_core_sdk.py` or new focused test file.

## Files/Areas Must Not Touch
- `benchmarks/**`
- `README.md`
- Docker files

## Architectural Constraints
- Keep deterministic behavior.
- Respect strategy order deterministically.
- Do not select duplicate sample ids.
- If pool size is smaller than `k`, return all possible ids.

## Step -> Verify Plan
- Update mix selection to track remaining ids -> verify each component sees reduced pool.
- Add allocation snapshot details -> verify state is JSON-serializable.
- Add tests for overlapping component outputs and fallback.
- Run focused tests and full pytest.

## Acceptance Criteria
- Mix mode no longer silently wastes allocation on already selected ids.
- Fallback cannot reselect ids already selected by components.
- Tests cover overlap and underfill.
- Full tests pass.

## Expected Tests and Validations
- `uv run --group dev pytest tests/test_core_sdk.py tests/test_strategy_correctness.py -q`
- `uv run --group dev pytest -q`

## Dependencies
- Can run independently from W14 because write scopes do not overlap.

## Parallel/Sequential Notes
- Do not run in parallel with any worker editing `engine.py`.
