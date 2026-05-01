# W21 - Make Benchmark IDs Truly Label-Independent

## Relation To Overall Task
R30 found that W20 removed class-name strings from acquisition-visible IDs/groups, but IDs/groups still encode labels through contiguous numeric ranges assigned inside label-ordered loops. This blocks accepting the benchmark harness.

## Assumptions And Resolved Ambiguities
- R30 finding is accepted as valid.
- Acquisition-visible IDs and group IDs must not reveal labels directly or through obvious contiguous ranges.
- It is acceptable for the synthetic text content to be class-informative.
- Private `BenchmarkSample.label` may be used for training, evaluation, diagnostics, and artifact label distributions after acquisition.

## Goal And Expected Result
Refactor synthetic dataset construction so all acquisition-visible sample IDs and group IDs are assigned in a label-independent deterministic way. Smoke artifacts must be regenerated.

## Responsibility Boundaries
Owned by this worker:
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md` if needed
- `benchmarks/results/smoke/**`

Do not change:
- `src/active_learning_sdk/**`
- `pyproject.toml`
- root `README.md`
- `docs/**`
- unrelated task docs

## In Scope
- Assign opaque sample IDs after deterministic shuffling across labels within each split, or use stable label-independent hashes/counters over shuffled records.
- Assign opaque group IDs so group ranges do not reveal labels. Preserve duplicate grouping for `grouped_duplicates`.
- Add or run validation that no simple contiguous ID/group ordering maps to label blocks. At minimum, inspect sorted acquisition-visible IDs by split and assert labels are mixed for datasets with multiple labels.
- Regenerate smoke artifacts.

## Out Of Scope
- SDK algorithm changes.
- External datasets.
- Timing split/performance optional cleanup.

## Files Or Modules May Be Changed
- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md`
- `benchmarks/results/smoke/**`

## Files Or Areas Must Not Be Touched
- `src/active_learning_sdk/**`
- `pyproject.toml`
- root `README.md`
- `docs/**`
- `.git/**`

## Important Architectural Constraints And Forbidden Actions
- Do not merely rename labels or shift numeric ranges.
- Do not expose labels in `DataSample.meta`, `sample_id`, `group_id`, schema, or predictable label-ordered ranges.
- Do not break duplicate/group diagnostics; group IDs should still identify near-duplicate groups, just opaquely.
- Do not reintroduce notebooks.

## High-Level Execution Plan
- Create helper(s) that build label-bearing internal records first, then assign opaque IDs/groups after deterministic split-level shuffle.
- For grouped duplicate data, preserve internal group membership while mapping internal groups to opaque IDs after a label-independent shuffle.
- Update smoke artifacts.
- Validate strict JSON and no label leakage through names/ranges.

## Step -> Verify Plan
- Refactor builders -> verify all sample IDs/groups are opaque and stable.
- Validate sorted IDs do not form single-label blocks -> run a Python assertion over every dataset/split.
- Run smoke benchmark -> verify 30 metrics rows and 30 selection rows.
- Verify strict JSON/no NaN remains true.

## Acceptance Criteria
- R30 P1 is fixed: acquisition-visible IDs/groups are label-independent beyond the unavoidable information in text/model predictions.
- Smoke artifacts are regenerated.
- Strict JSON remains valid.
- No SDK files changed.

## Expected Tests And Validations
- `python -m py_compile benchmarks/sdk_first_benchmark.py`
- `python benchmarks/sdk_first_benchmark.py --help`
- `.\\.venv\\Scripts\\python.exe benchmarks/sdk_first_benchmark.py --preset smoke`
- Python validation for no labels in acquisition-visible schema/id/group/meta.
- Python validation that sorted sample IDs and sorted group IDs are not label-block ordered for multi-label splits.
- Strict JSON parse/no NaN check.

## Dependencies
Depends on W20 and R30.

## Parallel Or Sequential Notes
Sequential gate before accepting benchmark harness.
