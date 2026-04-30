# Review Task: wave3-api

## Context
Review Wave3 API regression/new-issue findings after supposed fixes. SDK source inspection remains forbidden.

## Goal
Validate `.agents/tmp/blackbox_stress/wave3_api_findings.md`, especially whether old issues are truly fixed in the black-box run and whether the new `get_round` raw `KeyError` finding is real.

## Responsibility Boundaries
Owns only `.agents/tmp/blackbox_stress/reviews/wave3_api_review.md`.

## In Scope
- Read public docs, prior reports, Wave3 task, harness, raw JSON/CSV, and findings.
- Rerun the Wave3 API harness or targeted cases if needed.
- Mark old-finding regression states and new findings accepted/rejected/needs-more-evidence.

## Out of Scope
- Reading `src/active_learning_sdk/**`.
- Modifying SDK source/tests/docs/benchmarks or worker artifacts.

## Acceptance Criteria
Review note gives a verdict on the old-regression table and each new material finding.
