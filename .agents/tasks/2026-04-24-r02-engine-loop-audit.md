# Task ID: 2026-04-24-r02-engine-loop-audit

## Relation To Overall Task

This is the SDK loop semantics research slice. It identifies current engine behavior that can distort active learning quality, especially SDK vs local benchmark differences.

## Goal

Produce a read-only audit of engine behavior that affects learning quality, budget fairness, resume semantics, training data composition, and metric interpretation.

## Responsibility Boundaries

Read-only. Do not edit files.

## In Scope

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/annotation.py`
- `src/active_learning_sdk/cache.py`
- state persistence files if needed for behavior understanding
- SDK-path logic in benchmark runner/notebooks

## Out Of Scope

- Label Studio HTTP details unless they affect learning loop semantics;
- implementing fixes;
- README edits.

## Files That May Be Changed

None.

## Files That Must Not Be Touched

The entire repository. Review only.

## Architectural Constraints

- Focus on whether comparisons are fair under equal label budgets.
- Look for mismatches between SDK and local benchmark loops.
- Identify things that can change selected examples, labels, rounds, model fit calls, or metrics.

## Execution Plan

1. Trace one SDK run round-by-round.
2. Compare SDK loop with notebook-local loop.
3. Identify metric or budget mismatches.
4. Identify hidden behavior that can affect selection quality.
5. Propose measurable improvements and tests.

## Acceptance Criteria

- Output lists all engine/loop surfaces worth attention.
- Output identifies likely causes of current SDK underperformance if visible.
- Each proposed improvement has a metric and a test/validation suggestion.
