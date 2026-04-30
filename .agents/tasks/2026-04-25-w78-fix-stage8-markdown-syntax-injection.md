# W78 - Fix Stage 8 Markdown Syntax Injection

## Context

R96 found that Markdown report values still allow Markdown syntax injection such as `[click](javascript:...)` and image syntax even after raw HTML escaping.

## Goal

Treat all user-controlled Markdown values as plain text by escaping Markdown control characters in addition to HTML escaping.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/report.py`
- `tests/test_report_generation.py`

## Acceptance Criteria

- Markdown report output does not render injected links/images/emphasis from user-controlled values.
- Existing HTML escaping remains intact.
- Tests and full suite pass.

## Validation

- `uv run --group dev pytest -q tests/test_report_generation.py`
- `uv run --group dev pytest -q`
