# Review Task: wave4-api-cache

## Context
Review Wave4 API/cache regression findings. SDK source inspection remains forbidden.

## Goal
Validate `.agents/tmp/blackbox_stress/wave4_api_cache_findings.md`, especially old issue closure and the new mutable `get_state()` finding.

## Responsibility Boundaries
Owns only `.agents/tmp/blackbox_stress/reviews/wave4_api_cache_review.md`.

## In Scope
- Read public docs, prior reports, Wave4 task, harness/results/findings.
- Rerun targeted public-runtime probe if needed.
- Mark findings accepted/rejected/needs-more-evidence and caveats.

## Out of Scope
- Reading `src/active_learning_sdk/**`.
- Modifying SDK or worker artifacts.

## Acceptance Criteria
Review note gives verdict on old/open regressions and new `get_state()` issue.
