# 2026-04-27-r129 Review Public API Audit

## Context

Review of the parent-orchestrator-added public API acceptance test from the 2026-04-27 senior audit.

## Goal

Independently verify whether `tests/test_senior_audit_public_api_2026_04_27.py` exposes a real public API defect around documented custom selector mode.

## Responsibility Boundaries

May inspect all repository files. May change only:

- `.agents/tmp/2026-04-27-r129-review-public-api.md`

Must not edit tests or SDK implementation.

## In Scope

- `tests/test_senior_audit_public_api_2026_04_27.py`
- `src/active_learning_sdk/configs.py`
- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- existing custom selector tests

## Review Questions

- Is `SchedulerConfig(mode="custom", custom_selector=...)` advertised/accepted by the public config?
- Does project configuration currently fail before the selector can run?
- Is the failing test a fair acceptance test or an overreach?
- What correction, if any, is needed for final reporting?

## Acceptance Criteria

Write a concise review note with verdict, commands run, and confirmed/rejected finding.
