# Task: blackbox-stress-api-state

## Context
The user requested an aggressive black-box stress test of the Active Learning SDK. Source-code inspection is forbidden. Only documentation may be read, plus public runtime behavior through importing and using the installed/editable SDK.

## Goal
Find public API, state, resume, cache, split, annotation, report, and error-taxonomy problems by writing and running external consumer-style stress tests.

## Responsibility Boundaries
Owns only artifacts under `.agents/tmp/blackbox_stress/api_state/` and notes in `.agents/tmp/blackbox_stress/api_state_findings.md`.

## In Scope
- Use `README.md`, `docs/README.md`, `docs/SDK_CONTRACTS.md`, and benchmark docs as API references.
- Import `active_learning_sdk` and public/provisional modules exactly as a user would.
- Create external test scripts, generated datasets, fake models, fake backends, and temporary project workdirs.
- Attack edge cases: duplicate IDs, missing text, invalid probabilities, label mismatch, empty pools, resume with changed data, corrupted state files, repeated close/run calls, report/export behavior, cache boundaries, timeout policies, custom selector signatures.

## Out of Scope
- Reading `src/active_learning_sdk/**` source files.
- Modifying SDK source, tests, benchmarks, docs, or repo configuration.
- Fixing bugs.
- Using private attributes as test dependencies, except observing serialized public artifacts in temp workdirs.

## Must Not Touch
- `src/**`
- `tests/**`
- `benchmarks/**` except running documented commands if useful
- existing `.agents/tasks/**` other than this document

## Architectural Constraints
Treat the SDK as an installed black box. Failures must be reproducible from public API calls and documented contracts.

## Execution Plan
1. Build a small black-box harness in `.agents/tmp/blackbox_stress/api_state/`.
2. Run focused stress cases with clear PASS/FAIL/ISSUE output.
3. Classify each issue with severity, reproduction command, observed behavior, expected behavior from docs, and affected public contract.

## Acceptance Criteria
- At least 20 distinct cases attempted.
- Findings include exact command(s), artifact paths, and a terse severity ranking.
- Explicitly state if a case could not be run and why.

## Dependencies
Can run in parallel with benchmark/model stress tasks. No shared write scope.
