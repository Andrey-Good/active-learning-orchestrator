# 2026-04-30 Wave7 Black-Box SDK Stress Master Plan

## Task Identifier

WAVE7-BLACKBOX-STRESS - documentation-only hostile SDK stress campaign.

## Context

The user requested a broad, harsh SDK stress test. Inspecting SDK implementation source is forbidden. Public documentation, package metadata, generated artifacts, and black-box SDK execution are allowed. Prior black-box waves exist and may be used only as historical context; Wave7 must produce fresh evidence.

## Goal

Find reproducible SDK defects, weak metrics, misleading documentation, missing validation, packaging failures, and operational issues using the SDK as a user-facing dependency.

## Decomposition Answers

1. This can be split cleanly by responsibility because API/model fuzzing, benchmark quality, packaging, and backend/report behavior have separate harnesses and output directories.
2. W07A, W07B, W07C, and W07D can run in parallel because their write scopes do not overlap.
3. Reviewer tasks must run after their assigned workers. Final synthesis must run after all workers/reviewers complete.

## Global Rules

Allowed:
- Read `README.md`, `docs/**`, `benchmarks/README.md`, `pyproject.toml`, task docs, and generated artifacts.
- Run documented benchmark commands without reading benchmark implementation source.
- Import and execute public `active_learning_sdk` APIs.
- Write new scripts, logs, results, and reports only under `.agents/tmp/blackbox_stress_wave7/`.
- Download bounded public datasets/models when useful.

Forbidden:
- Read or inspect `src/active_learning_sdk/**`.
- Read repository `tests/**` as hints.
- Read benchmark implementation source such as `benchmarks/*.py`; running documented benchmark commands is allowed.
- Modify SDK source, existing tests, benchmark source, package metadata, lockfiles, or existing docs.
- Revert unrelated dirty worktree changes.
- Use private SDK attributes or internals.
- Report a problem without a command/script, observed behavior, expected behavior from docs/contracts, severity, and reproduction notes.

## Subtasks

- W07A: API/model contract fuzz and adversarial runtime behavior.
- W07B: documented benchmark/quality stress across synthetic and real datasets.
- W07C: packaging, optional extras, root imports, README quickstarts, and wheel/sdist smoke.
- W07D: backend, report, export, timeout, and operational contract stress.

## Acceptance Criteria

- Every worker writes `findings.md` and machine-readable results under its assigned directory.
- Every finding is either confirmed by reviewer or explicitly downgraded/rejected.
- Metric claims use matched random baselines or are labeled as diagnostics.
- Final report is written to `.agents/tmp/blackbox_stress_wave7/final_report.md`.

## Dependencies

Workers W07A-W07D run in parallel. Reviews run after workers. Final synthesis runs after reviews.
