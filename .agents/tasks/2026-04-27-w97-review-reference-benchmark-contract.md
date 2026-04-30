# Task W97-K: Review Reference Benchmark P1 Fix

## Context
Task W97-I changed `benchmarks/reference_strategy_benchmark.py`, `tests/test_reference_strategy_benchmark.py`, and `benchmarks/README.md` after final review found P1 blockers:
- reference/manual formulas normalized invalid probability rows;
- reference benchmark could overwrite evidence and lacked reproducibility metadata.

## Goal
Perform a read-only senior review of the W97-I fix.

## Scope
Read only:
- `benchmarks/reference_strategy_benchmark.py`
- `tests/test_reference_strategy_benchmark.py`
- `benchmarks/README.md`
- relevant helper functions in `benchmarks/sdk_first_benchmark.py`

Do not edit files.

## Review Questions
- Does the reference benchmark now reject count-like/logit-like/non-finite/negative/wrong-width probability rows instead of normalizing them?
- Is the strict probability contract aligned with SDK uncertainty strategy behavior?
- Does default output avoid overwriting previous evidence?
- Does explicit output-dir no-clobber unless `--overwrite` work through existing helper semantics?
- Does the manifest include argv, git, runtime, artifact schema, and artifact filenames?
- Are tests meaningful and not just superficial?
- Are docs honest and current?

## Output
Return findings ordered by severity. If no release blockers remain in this scope, say so clearly and list residual risks.
