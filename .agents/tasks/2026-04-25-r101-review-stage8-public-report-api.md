# R101 - Review Stage 8 Public Report API Tail

## Context

Stage 8 added dependency-free JSON/Markdown/HTML project reports, a reproducibility manifest, and strict persisted-state JSON safety. After the previous clean review, a small public API gap was closed: `ActiveLearningEngine.generate_report()` and `ActiveLearningProject.generate_report()` now return the artifact path dictionary produced by `ReportGenerator.generate_report()`.

## Goal

Review the final Stage 8 public report API tail for correctness, compatibility, and product quality.

## Responsibility Boundaries

You are a reviewer only. Do not edit files.

## In Scope

- `src/active_learning_sdk/engine.py`
- `src/active_learning_sdk/project.py`
- `src/active_learning_sdk/report.py`
- `tests/test_report_generation.py`
- Relevant report/state tests if needed

## Out of Scope

- Benchmark implementation
- README edits
- Docker/Label Studio changes
- Strategy algorithm changes

## Special Attention

- Public API should return useful artifact paths without breaking report generation.
- Path behavior should remain compatible with directory and file-style outputs.
- Tests should pin the new return contract adequately.
- The implementation must not mutate report state beyond existing behavior.

## Forbidden Actions

- Do not modify source files.
- Do not revert unrelated workspace changes.
- Do not expand scope into Stage 9.

## Review Plan

1. Inspect the public `generate_report` wrappers and report generator path contract.
2. Inspect report tests covering the new return value.
3. Run targeted tests if helpful.
4. Report findings with severity and exact file/line references, or state that no blocking findings remain.

## Acceptance Criteria

- No correctness regressions in public report generation.
- Return type and path values are coherent.
- Tests pass or any failure is clearly explained.

## Dependencies

Depends on W76-W81 and the direct public API tail patch.
