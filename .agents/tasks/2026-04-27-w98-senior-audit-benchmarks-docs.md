# 2026-04-27-w98 Senior Audit Benchmarks/Docs

## Context

Part of the senior acceptance audit requested on 2026-04-27. The user explicitly requested benchmarks showing where this SDK is worse than analogs or direct manual work, plus a text file listing code-review objections.

## Goal

Audit benchmark validity, packaging/documentation honesty, and comparison claims. Add or update benchmark evidence in an owned result directory and record precise findings for final integration.

## Responsibility Boundaries

May inspect all repository files. May change only:

- `benchmarks/results/senior_audit_2026_04_27/**`
- `.agents/tmp/2026-04-27-w98-benchmarks-docs-notes.md`

Must not edit SDK implementation files, tests, benchmark source scripts, README, or final docs.

## In Scope

- `benchmarks/*`
- existing benchmark result artifacts
- README/docs benchmark claims
- package/build metadata and dependencies only as audit targets
- direct SDK-vs-manual benchmark runs using existing harness

## Out of Scope

- Implementing new benchmark harness code.
- Fixing benchmark scripts.
- Runtime/strategy source changes.

## Constraints

- Do not overwrite existing benchmark result directories.
- Use existing benchmark primitives where possible.
- Clearly distinguish measured results from limitations and unmeasured analogs.
- If external analogs are unavailable locally, report that honestly instead of simulating results.

## Execution Plan

1. Inspect benchmark scripts, existing outputs, and README/docs claims.
2. Run a fresh SDK-vs-manual benchmark into `benchmarks/results/senior_audit_2026_04_27`.
3. Validate artifact consistency.
4. Write notes to `.agents/tmp/2026-04-27-w98-benchmarks-docs-notes.md`.

## Acceptance Criteria

- Fresh benchmark artifacts exist in the owned result directory.
- Notes include measured overhead/quality deltas and any claim-validity issues.
- Commands and outcomes are included.

## Dependencies

Runs in parallel with runtime and strategy audit tasks. No shared write scope.
