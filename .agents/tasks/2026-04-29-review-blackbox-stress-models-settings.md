# Review Task: blackbox-stress-models-settings

## Context
Review the model/settings black-box stress findings produced for the user's SDK stress test. Source-code inspection remains forbidden.

## Goal
Validate `.agents/tmp/blackbox_stress/model_settings_findings.md` for reproducibility and fair interpretation of public adapter/scheduler contracts.

## Responsibility Boundaries
Owns only notes under `.agents/tmp/blackbox_stress/reviews/model_settings_review.md`.

## In Scope
- Read the model/settings task doc, generated harness, findings, public documentation, and result JSONL/summary.
- Rerun the reproduction command or targeted cases if needed.
- Verify whether stochastic/committee shape ambiguity is truly a documentation/API problem.

## Out of Scope
- Reading `src/active_learning_sdk/**`.
- Modifying SDK source or worker harness.
- Fixing bugs.

## Must Not Touch
- `src/**`
- worker-owned artifacts except by reading/rerunning

## Acceptance Criteria
The review note marks each finding as accepted, rejected, or needs-more-evidence, with rationale and any reproduction command used.
