# 2026-04-29 Wave6 Black-Box SDK Stress Master Plan

## Task Identifier

WAVE6-BLACKBOX-STRESS - documentation-only expanded SDK stress campaign.

## Context

The user requested a harsh, broad stress test of the SDK. The SDK implementation source is forbidden. Prior black-box waves already covered many public API, state/cache, packaging, and benchmark cases, but the residual-risk list still includes DataFrame/CSV/Parquet ingestion, broader real-dataset quality, and stronger persistent-cache/load probes.

## Goal

Find new reproducible SDK defects, weak metrics, missing guardrails, and documentation mismatches by using the SDK strictly as a public black-box dependency.

## Responsibility Boundaries

Allowed:
- Read public documentation: `README.md`, `docs/**`, task documents, generated reports, and package metadata needed for install commands.
- Run documented benchmark commands without inspecting benchmark implementation source.
- Write new scripts, outputs, and reports under `.agents/tmp/blackbox_stress_wave6/`.
- Import and execute `active_learning_sdk` public APIs as a normal user.
- Download bounded public datasets/models when useful.

Forbidden:
- Read or inspect `src/active_learning_sdk/**`.
- Read existing repository `tests/**` as hints.
- Modify SDK source, existing tests, package metadata, lockfiles, or existing benchmark source.
- Revert unrelated dirty worktree changes.
- Use private SDK attributes or internals.

## Decomposition

Parallel worker subtasks:
- W06A: public ingestion/export stress for provider, DataFrame, CSV, and Parquet surfaces.
- W06B: persistent cache, filesystem, process, and resume stress.
- W06C: documented real/synthetic quality matrix with multiple datasets, strategies, seeds, and small/medium budgets.

Sequential review subtasks:
- R06A reviews W06A and W06B reproducibility and severity.
- R06B reviews W06C metrics and rejects benchmark false positives.
- Final system-level synthesis after workers and reviewers complete.

## Acceptance Criteria

- Every confirmed finding names expected behavior from docs/contracts, observed behavior, severity, command, artifact path, and reproduction notes.
- Metric claims use matched random baselines or are clearly labeled as diagnostics.
- Known historical defects are not double-counted unless they still reproduce.
- Final report is written to `.agents/tmp/blackbox_stress_wave6/final_report.md`.

## Dependencies

W06A/W06B/W06C may run in parallel. Reviews wait for their assigned workers. Final synthesis waits for reviews.
