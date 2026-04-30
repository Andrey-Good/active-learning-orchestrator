# W77 - Fix Stage 8 Markdown Escaping

## Context

R95 found that Markdown report values can render raw HTML because `_md()` escapes only table pipes/newlines and the report title bypasses `_md()`.

## Goal

Escape Markdown report values so user-controlled project names, labels, sample ids, and payloads cannot render as raw HTML in common Markdown renderers.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/report.py`
- `tests/test_report_generation.py`

Do not edit engine/report API plumbing.

## Acceptance Criteria

- Markdown title and table/code values do not emit raw `<script>`, `<img>`, `<svg>`, etc. from user-controlled data.
- Existing JSON semantics remain unchanged.
- HTML artifact remains escaped.
- Report tests and full suite pass.

## Validation

- `uv run --group dev pytest -q tests/test_report_generation.py`
- `uv run --group dev pytest -q`
