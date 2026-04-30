# Task W97-I: Reference Benchmark Contract And Artifact Hygiene

## Context
Final independent review rejected release because `benchmarks/reference_strategy_benchmark.py` still has stale behavior after the SDK strict probability contract was fixed.

## Goal
Make the reference benchmark honest and safe:
- manual/reference uncertainty formulas must reject invalid probability rows exactly like the SDK strict contract;
- benchmark output must not overwrite evidence accidentally;
- manifest/docs/tests must describe the current behavior.

## Ownership
May change:
- `benchmarks/reference_strategy_benchmark.py`
- `tests/test_reference_strategy_benchmark.py`
- `benchmarks/README.md`
- narrowly related README benchmark text if necessary

Must not change:
- SDK runtime/strategy/backend code
- unrelated benchmark harnesses unless a direct compatibility edit is required

## Scope
In scope:
- Replace positive-row normalization with strict row validation: 2D-like row, at least two columns, finite, non-negative, sums to 1.0 within SDK tolerance.
- Add/adjust tests proving count-like/logit-like rows are rejected and valid rows pass unchanged.
- Add no-clobber behavior for explicit/default output directories and an explicit `--overwrite` option.
- Add reproducibility metadata compatible with the SDK-first benchmark manifest: argv, git sha/dirty count, runtime, artifact schema/version, artifact filenames.
- Update docs to remove stale normalization claims.

Out of scope:
- Rewriting all benchmark logic.
- Adding external `modAL`/`skactiveml` dependencies.

## Acceptance Criteria
- Focused reference benchmark tests pass.
- Re-running the reference benchmark against an existing non-empty output dir fails unless `--overwrite` is supplied.
- Fresh reference smoke benchmark completes.
- Documentation no longer claims manual probability rows normalize invalid inputs.
