# W12 - SDK Strategy Correctness

## Relation to Overall Task
Correctness-level improvement for existing SDK strategies. This should make selection deterministic, validated, and safer before quality-tuning strategy behavior.

## Assumptions and Resolved Ambiguities
- Do not change benchmark runner in this task.
- Do not add new strategy names.
- This task improves existing random/entropy/margin/least-confidence behavior.
- Quality lift is not required from this task; reproducibility and safe failure are.

## Goal
Implement robust probability validation/normalization and deterministic tie-breaking for built-in strategies.

## Responsibility Boundaries
- Own only SDK strategies and focused tests.

## In Scope
- `src/active_learning_sdk/strategies/uncertainty.py`
- tests under `tests/`
- Add small helper functions if needed in the same module.
- Validate `predict_proba` output:
  - one probability row per pool id;
  - non-empty rows;
  - finite numeric values;
  - non-negative values;
  - positive row sum;
  - normalize rows if they do not sum exactly to one within tolerance.
- Deterministic tie-break:
  - stable across runs given same input order;
  - not biased purely by original pool order when scores tie;
  - use stable hash/sample id/model id/strategy name, not global RNG.
- `RandomStrategy` should avoid global RNG and be deterministic from context/model/pool ids.

## Out of Scope
- Do not edit `engine.py`.
- Do not change scheduler mix behavior.
- Do not add new acquisition algorithms.
- Do not touch benchmark artifacts.

## Files/Modules May Change
- `src/active_learning_sdk/strategies/uncertainty.py`
- `tests/test_core_sdk.py` or a new focused test file under `tests/`

## Files/Areas Must Not Touch
- `benchmarks/**`
- `README.md`
- Docker files

## Architectural Constraints
- Keep the public strategy API unchanged.
- Fail with `ConfigurationError` for invalid probability output.
- Preserve deterministic selection for reproducibility.

## Step -> Verify Plan
- Add probability helper -> tests invalid rows and normalization.
- Add deterministic tie ordering helper -> tests equal-score strategies do not simply return first pool ids and are stable.
- Update RandomStrategy -> tests repeated calls match and do not mutate global RNG.
- Run focused tests -> run full pytest.

## Acceptance Criteria
- Built-in strategies validate malformed probabilities.
- Equal-score uncertainty selections are stable but not pool-order-first.
- Random selection is deterministic for the same context/pool/model id.
- All tests pass.

## Expected Tests and Validations
- `uv run --group dev pytest tests/test_core_sdk.py -q`
- `uv run --group dev pytest -q`

## Dependencies
- Can run in parallel with R19.

## Parallel/Sequential Notes
- Do not edit `engine.py`; scheduler mix gets a separate task if needed.
