# 2026-04-26-r124 Repeat Acceptance: Benchmarks And Docs

## Context

The user says the previous senior-review blockers were fixed and requests a repeat SDK acceptance verdict.

Previous benchmark/docs blockers:

- manual-vs-SDK fixture did not distinguish entropy, margin, and least-confidence;
- benchmark reruns could overwrite evidence;
- README and audit docs had stale or corrupted acceptance evidence.

## Goal

Verify whether benchmark evidence is now honest enough for SDK acceptance and whether docs are internally consistent.

## Responsibility Boundaries

In scope:

- `benchmarks/audit_sdk_vs_manual.py`
- `benchmarks/results/*`
- `tests/test_audit_benchmark_comparison.py`
- `tests/test_senior_acceptance_blockers.py`
- `README.md`
- `docs/*audit*`

Out of scope:

- Editing files.
- Implementing external benchmark adapters.
- Runtime/strategy correctness except where it affects benchmark claims.

## Prohibitions

- Do not modify files.
- Do not revert user changes.
- Do not fabricate external analog results.

## Plan

1. Inspect benchmark harness and docs.
2. Run benchmark tests and a fresh non-overwriting benchmark if feasible.
3. Verify fixture discriminates strategies and artifacts are not overwritten accidentally.
4. Return verdict, findings, commands/results, and remaining evidence limits.

## Acceptance Criteria

- Clear accepted/not accepted benchmark/docs verdict.
- Honest statement of what current benchmark does and does not prove.
- Source-grounded findings if any.
