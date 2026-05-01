# Task ID: 2026-04-24-r01-strategy-scheduler-audit

## Relation To Overall Task

This is the strategy/scheduler research slice for the scientific optimization program. It identifies everything in current acquisition logic that can affect active learning quality before we add new large methods.

## Goal

Produce a read-only audit of current strategy and scheduler behavior, with concrete improvement targets and suggested metrics.

## Responsibility Boundaries

Read-only. Do not edit files.

## In Scope

- `src/active_learning_sdk/strategies/**`
- scheduler-related parts of `src/active_learning_sdk/engine.py`
- `SchedulerConfig` in `src/active_learning_sdk/configs.py`
- strategy usage in notebooks/benchmarks
- tests covering strategy behavior

## Out Of Scope

- implementing new strategies;
- editing benchmark artifacts;
- README edits;
- Label Studio backend behavior unless it directly affects selection.

## Files That May Be Changed

None.

## Files That Must Not Be Touched

The entire repository. Review only.

## Architectural Constraints

- Treat active learning as budget-limited.
- Separate strategy quality from model training quality.
- Look for edge cases: ties, class imbalance, early cold-start, duplicate batch selections, pool exhaustion, `k > pool`, missing probabilities.

## Execution Plan

1. Read current strategy implementations.
2. Read scheduler selection behavior.
3. Read tests and benchmark usage.
4. List all quality-affecting behavior.
5. Propose metrics for each behavior.
6. Propose hypotheses to test.

## Acceptance Criteria

- Output includes a prioritized list of strategy/scheduler improvement targets.
- Each target has at least one measurable metric.
- Output distinguishes low-risk fixes from method additions.
- Output explicitly states whether BADGE/diversity are absent or placeholders.
