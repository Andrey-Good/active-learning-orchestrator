# w89 - Fix Benchmark Validity Issues

## Context

The final review found two benchmark-validity issues:

- capped real datasets can keep labels in the label schema that were only present in discarded rows;
- selected embedding redundancy diagnostics are computed after retraining on the selected labels instead of acquisition-time model state.

## Goal

Make capped real-dataset labels reflect retained samples and compute selection embedding diagnostics from the acquisition-visible model state.

## Responsibility Boundaries

Own only:

- `benchmarks/sdk_first_benchmark.py`
- benchmark-specific tests in `tests/test_sdk_first_benchmark_embedding_diagnostics.py` or a new benchmark test file

Do not edit quality gate code, README, SDK strategy implementations, or engine capability validation.

## In Scope

- In real dataset construction, apply train/test caps before computing `labels`/`labels_seen` for the returned `BenchmarkDataset`.
- Preserve deterministic ordering and existing split behavior.
- Ensure full-train reference and coverage metrics use the corrected labels.
- In `run_one_curve`, capture selected embeddings for diagnostics before adding selected ids/retraining, using the model state visible at acquisition time.
- Preserve selected IDs, metrics, and selection behavior.
- Add tests that would fail under the old behavior.

## Out Of Scope

- No dataset downloads or long benchmark runs.
- No strategy tuning.
- No quality gate threshold changes.

## Constraints

- Do not leak oracle labels into acquisition selection.
- Keep benchmark artifacts strict JSON/CSV compatible.
- Avoid extra model training work.

## Acceptance Criteria

- Tests prove capped real labels are recomputed from retained samples.
- Tests prove selected embedding diagnostics use acquisition-time embeddings.
- `uv run pytest tests/test_sdk_first_benchmark_embedding_diagnostics.py -q` passes.

## Dependencies

None.
