# Task W96 Review: Senior Audit Blocker Fixes

## Context
The user provided a senior audit report dated 2026-04-27 with five release-blocking issues. A worker already fixed the `LLMLabelBackend` payload-preservation slice. The main implementation then fixed strict state validation, prediction-cache key aliasing, `validate()` task-id checks, and JSON-safe persistence of runtime-only custom selector callables.

## Goal
Review the current code changes for correctness, maintainability, and hidden regressions. This is a read-only review task unless explicitly asked to patch after reporting findings.

## Responsibility Boundaries
In scope:
- Review changes in `src/active_learning_sdk/cache.py`.
- Review changes in `src/active_learning_sdk/state/store.py`.
- Review changes in `src/active_learning_sdk/engine.py`.
- Review changes in `src/active_learning_sdk/backends/base.py`.
- Review the three audit test files from 2026-04-27.

Out of scope:
- Do not modify benchmark artifacts, README, docs, notebooks, Docker files, or unrelated SDK features.
- Do not weaken, delete, or xfail tests.
- Do not rewrite unrelated architecture.

## Acceptance Criteria
- The five audit blockers are actually fixed, not merely hidden from tests.
- Existing public behavior is preserved where possible.
- Runtime-only custom selector behavior is explicit and safe.
- Cache key scoping cannot alias when model/sample ids contain delimiters.
- State loading rejects invalid `sample_status` values deterministically.
- `project.validate()` reports mismatched `selected_sample_ids` and `task_ids`.
- `LLMLabelBackend` passes original `DataSample` payloads into `label_fn`.

## Expected Validation
- Inspect the relevant code paths.
- Optionally run focused tests if needed.
- Report concise findings with file/line references and severity.
- If no blockers are found, say so explicitly and note residual risks.
