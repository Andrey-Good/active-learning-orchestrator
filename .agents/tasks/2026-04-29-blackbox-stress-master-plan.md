# 2026-04-29 Blackbox SDK Stress Master Plan

## Task Identifier

BLACKBOX-STRESS-MASTER - documentation-only SDK stress campaign.

## Context

The user requested an aggressive stress test of the SDK while forbidding inspection of SDK implementation code. Documentation may be read. Test code, scripts, datasets, virtual environments, and generated reports may be created freely.

## Goal

Find reproducible SDK failures, weak metrics, misleading behavior, missing guardrails, bad error handling, packaging issues, and evidence gaps using only public documentation and black-box SDK usage.

## Boundaries

Allowed:
- Read `README.md`, files under `docs/`, benchmark documentation, packaging metadata only when needed for install commands, and generated black-box artifacts.
- Import and execute `active_learning_sdk` as a user would.
- Write new stress scripts, output artifacts, logs, and reports under `.agents/tmp/blackbox_stress_2026_04_29/`.
- Download public datasets/models if useful and bounded.

Forbidden:
- Read or inspect `src/active_learning_sdk/**`.
- Read existing `tests/**` as implementation hints.
- Modify SDK source, existing tests, docs, benchmarks, lockfiles, or package metadata.
- Revert or clean unrelated dirty worktree changes.
- Use private attributes or internals discovered from source code.

## Decomposition

Parallel worker subtasks:
- W01 public API contract and adversarial model/dataset/backends.
- W02 active-learning quality stress across synthetic and real/capped datasets.
- W03 state, resume, cache, filesystem, and concurrent-process stress.
- W04 packaging, extras, docs quickstarts, and optional integration smoke.

Sequential review subtasks:
- R01 review W01/W03 reproducibility and severity.
- R02 review W02/W04 metrics, evidence categories, and false positives.
- FINAL system-level synthesis after worker and reviewer results.

## Acceptance Criteria

- Every finding includes a command or script path, environment assumptions, observed behavior, expected behavior from docs, severity, and reproduction notes.
- Metrics findings compare against matched baselines where applicable.
- False positives caused by violating documented preconditions are rejected or explicitly downgraded.
- Final report is written under `.agents/tmp/blackbox_stress_2026_04_29/final_report.md`.

## Dependencies

Workers may run in parallel. Reviewers must wait for worker artifacts. Final synthesis must wait for reviews.
