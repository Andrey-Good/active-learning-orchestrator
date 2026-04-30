# Task wave3-cache-semantics-research

## Context
Wave3 black-box testing says persistent cache no longer raises Windows `PermissionError`, but observability remains unacceptable: prediction-heavy runs show `writes > 0` while `items=0`, `data_bytes=0`, `index_bytes=0`, and stats reset after reopen.

## Goal
Research and propose the correct cache semantics and implementation fix. Do not edit files.

## In Scope
- Inspect cache store, engine automatic invalidation, `cache_stats()`, and wave3 load harness behavior.
- Determine whether current `items=0` is due automatic model-id invalidation after training.
- Recommend a release-quality public stats contract that distinguishes current reusable entries from lifetime writes/clears/invalidation.
- Identify focused tests needed.

## Out Of Scope
- Do not modify files.
- Do not weaken tests or reports.

## Acceptance Criteria
- Root cause statement.
- Proposed fields/semantics for stats.
- List exact files/functions to change and tests to add.
