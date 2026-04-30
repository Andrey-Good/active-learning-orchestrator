# w91 - Fix Audit Runtime Contracts

## Context

The audit added failing runtime acceptance tests for state/data validation:

- duplicate IDs from custom providers collapse silently;
- explicit splits accept IDs not present in the dataset;
- backend labels can persist values outside `LabelSchema.labels`;
- cached `predict_proba` can crash with `KeyError` on row-count mismatch.

## Goal

Fix runtime/state validation so all audit runtime tests pass without weakening existing behavior.

## Ownership

You own:

- `src/active_learning_sdk/engine.py`
- runtime validation tests only if needed, primarily `tests/test_audit_runtime_edge_cases.py`

Do not touch strategy implementations or benchmark code.

## In Scope

- Materialize provider sample IDs once during configure/attach validation and reject duplicates with `ConfigurationError`.
- Validate explicit split IDs against the provider ID set.
- Reject or prevent persistence of backend-resolved labels outside `LabelSchema.labels`.
- Apply the same backend label validation in timeout acceptance paths.
- Validate `predict_proba` row counts before writing cache entries and before reconstructing ordered rows.
- Raise actionable SDK exceptions (`ConfigurationError` or `ModelAdapterError`), not incidental Python exceptions.

## Out Of Scope

- No strategy scheduler changes.
- No benchmark changes.
- No README changes.

## Acceptance Criteria

- `uv run pytest tests/test_audit_runtime_edge_cases.py -q` passes.
- Relevant existing runtime tests still pass.
- Invalid labels never enter `sample_labels`.

## Notes

Do not revert unrelated work. The repo is dirty from prior stages.
