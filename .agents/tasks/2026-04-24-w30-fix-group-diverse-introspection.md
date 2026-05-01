# W30 - Fix Group-Diverse Strategy Introspection

## Relation To Overall Task
R39 accepted `group_diverse_entropy` for benchmark integration, with one non-blocking SDK polish issue: scheduler instances do not include `GroupDiverseEntropyStrategy()` in their registered strategy list, so `available_strategies()` and diagnostics omit it.

## Assumptions And Resolved Ambiguities
- Selection works already through `_get_strategy()`.
- This is a small introspection/registration consistency fix.

## Goal And Expected Result
Include `GroupDiverseEntropyStrategy()` in the built-in scheduler registry wherever built-ins are instantiated, and add/adjust a test proving `available_strategies()` includes `group_diverse_entropy`.

## Responsibility Boundaries
Owned by this worker:
- `src/active_learning_sdk/engine.py`
- `tests/**` if needed

Do not change:
- `benchmarks/**`
- strategy logic
- docs/root README

## In Scope
- Add built-in to configured scheduler strategy lists in `configure` and `attach_runtime`.
- Test introspection.
- Run targeted/full tests.

## Out Of Scope
- Benchmark runs.
- New strategy behavior.

## Files Or Modules May Be Changed
- `src/active_learning_sdk/engine.py`
- `tests/**`

## Files Or Areas Must Not Be Touched
- `benchmarks/**`
- `README.md`
- `docs/**`
- `docker/**`

## Acceptance Criteria
- `available_strategies()` includes `group_diverse_entropy` for a scheduler configured through engine/project path or direct scheduler if applicable.
- Tests pass.

## Expected Tests And Validations
- `uv run --group dev pytest tests/test_group_diverse_strategy.py -q`
- `uv run --group dev pytest -q`

## Dependencies
Depends on R39.

## Parallel Or Sequential Notes
Sequential before benchmark integration for clean public surface.
