# W79 - Fix Stage 8 Markdown Line Separator Injection

## Context

R97 found that Markdown escaping normalizes `\n` but not carriage returns or other line separators, allowing user-controlled values to break Markdown structure.

## Goal

Normalize line separators before inserting user-controlled values into Markdown reports.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/report.py`
- `tests/test_report_generation.py`

## Acceptance Criteria

- `\r`, `\r\n`, Unicode line separator, and Unicode paragraph separator cannot split generated Markdown structures.
- Report tests and full suite pass.

## Validation

- `uv run --group dev pytest -q tests/test_report_generation.py`
- `uv run --group dev pytest -q`
