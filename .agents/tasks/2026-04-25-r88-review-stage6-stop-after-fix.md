# R88 - Review Stage 6 Stop Criteria After W69 Fix

## Context

Stage 6 adds production-grade stop criteria and budget optimization traces to the active learning engine. W68 implemented the core stop criteria. R87 found issues around missing stopped traces for exhausted pools, sparse acquisition-score windows, and discarded non-stop diagnostics. W69 fixed those issues.

## Goal

Review W69 and determine whether Stage 6 stop-criteria core is now correct enough to build benchmark/simulation wiring on top of it.

## Responsibility Boundaries

You are a reviewer. Do not edit files.

## In Scope

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/configs.py`
- `tests/test_stop_criteria.py`
- Any directly relevant existing tests needed to understand expected behavior.

## Out Of Scope

- Benchmark harness changes.
- Label backend changes.
- README or documentation edits.
- New feature design beyond identifying correctness gaps in the current stop criteria implementation.

## Required Review Questions

- Does `run()` persist a stopped `stop_trace` when `StopCriteriaReached` is raised, including exhausted unlabeled pools?
- Does acquisition-score convergence use only recent completed rounds and refuse to stop when required recent rounds are missing the configured score key?
- Do non-stop traces retain useful diagnostic observations for configured stop checks?
- Are min gates (`min_labeled`, `min_rounds`) applied consistently across all stop checks?
- Are label-distribution and calibration stabilization checks robust to missing or malformed metrics?
- Are edge cases covered by tests: empty unlabeled pool, sparse acquisition traces, non-stop diagnostics, max budget, metric plateau, no evaluation data?

## Explicitly Forbidden

- Do not modify code.
- Do not broaden the task into new Stage 6 features.
- Do not revert unrelated changes in the repository.

## Validation To Run

- `uv run --group dev pytest -q tests/test_stop_criteria.py`
- `uv run --group dev pytest -q`

## Output

Return findings ordered by severity with file/line references. If no issues remain, say so explicitly and include validation results.
