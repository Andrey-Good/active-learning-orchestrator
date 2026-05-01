# R19 - Quality Improvement Hypotheses

## Relation to Overall Task
Read-only research task to convert accepted diagnostics into concrete benchmark-only quality experiments.

## Assumptions and Resolved Ambiguities
- Warm-start was rejected as strict quality improvement.
- Current accepted diagnostics point to class skew as a main failure mode.
- We can test benchmark-only composite/variant strategies before promoting anything to SDK.

## Goal
Recommend the next 2-4 small quality experiments most likely to improve benchmark metrics.

## Responsibility Boundaries
- Read-only.
- Do not edit files.

## In Scope
- Inspect accepted artifacts:
  - `baseline_sweep_*`;
  - `acquisition_*`;
  - `warm_start_*`.
- Inspect benchmark strategy implementation.
- Propose candidate variants with metrics:
  - predicted-class-balanced entropy/margin/least-confidence;
  - score smoothing/temperature acquisition if feasible;
  - round-dependent random mix if not already falsified by warm-start;
  - tie jitter diagnostics.

## Out of Scope
- No code changes.
- No long training runs.
- Do not recommend BADGE as immediate next step.

## Acceptance Criteria
- Final report gives prioritized experiments, exact metrics, success/failure criteria, and small run matrix.
- Distinguish benchmark-only variants from SDK promotion candidates.

## Expected Validations
- Read-only evidence from artifact values.

## Dependencies
- Can run in parallel with W12.
