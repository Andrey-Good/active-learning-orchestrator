# 2026-04-29 W03 State Cache Resume Stress

## Task Identifier

W03 - State, cache, resume, and filesystem stress.

## Context

Part of BLACKBOX-STRESS-MASTER. The SDK promises persisted state, resume-safe rounds, dataset fingerprint checks, locks, caches, reports, and audit artifacts.

## Goal

Break or weaken persistence, resume, locking, cache invalidation, report generation, export correctness, and behavior under interrupted/reopened projects using only public SDK APIs.

## Responsibility Boundaries

Write scope:
- `.agents/tmp/blackbox_stress_2026_04_29/w03_state_cache/**`

Readable sources:
- `README.md`
- `docs/README.md`
- `docs/SDK_CONTRACTS.md`
- W03-generated artifacts

Must not read:
- `src/active_learning_sdk/**`
- `tests/**`
- benchmark implementation source

## In Scope

- Reopen project and call `attach_runtime(...)` if documented/available.
- Run with cache on/off, persist true/false, stable and changing model ids.
- Dataset fingerprint mismatch with same ids/different text/payload/group.
- Duplicate runs, interrupted workdirs, corrupted public artifact files, concurrent opens.
- Public `status`, `validate`, `list_rounds`, `get_round`, `generate_report`, `export_labels`, `export_dataset_split`, `cache_stats`, `clear_cache`.

## Out Of Scope

- Direct editing of SDK source.
- Private state schema assumptions unless observable via public artifacts.
- Multi-writer cache correctness beyond documented single-writer contract.

## Execution Plan

1. Build a black-box script that creates projects, runs one or more rounds, closes, reopens, and resumes.
2. Perturb dataset/model/cache settings and observe documented exceptions or validation failures.
3. Probe lock behavior with concurrent processes if feasible.
4. Generate reports and exports, then validate artifact presence and basic consistency.
5. Produce `findings.md` and `results.json`.

## Acceptance Criteria

- At least 20 persistence/cache/resume scenarios.
- Each finding includes expected behavior from docs and observed behavior.
- Concurrency claims are scoped to the documented lock/single-writer semantics.

## Parallelism

Can run in parallel with W01, W02, W04.
