# Review Task: wave4-explore

## Context
Review Wave4 exploratory adapter/backend findings. SDK source inspection remains forbidden.

## Goal
Validate `.agents/tmp/blackbox_stress/wave4_explore_findings.md`, including simulator duplicate push/annotation behavior, LLM placeholder surface, and hostile property capability inspection.

## Responsibility Boundaries
Owns only `.agents/tmp/blackbox_stress/reviews/wave4_explore_review.md`.

## In Scope
- Read public docs, task, harness/results/findings.
- Rerun targeted public-runtime probes if needed.
- Classify each candidate issue as accepted/rejected/needs-more-evidence.

## Out of Scope
- Reading `src/active_learning_sdk/**`.
- Modifying SDK or worker artifacts.

## Acceptance Criteria
Review note gives a verdict per candidate issue and caveats.
