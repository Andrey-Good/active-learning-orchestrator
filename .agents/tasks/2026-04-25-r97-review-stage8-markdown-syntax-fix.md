# R97 - Review W78 Markdown Syntax Injection Fix

## Context

R96 found Markdown syntax injection in report Markdown values. W78 escapes Markdown control characters in addition to HTML escaping and added regression tests.

## Goal

Verify Markdown report user-controlled values are treated as plain text without breaking JSON/HTML report behavior.

## Responsibility Boundaries

You are a reviewer. Do not edit files.

## In Scope

- `src/active_learning_sdk/report.py`
- `tests/test_report_generation.py`

## Validation To Run

- `uv run --group dev pytest -q tests/test_report_generation.py`
- `uv run --group dev pytest -q`

## Output

Return findings ordered by severity. If no findings remain, say so explicitly and include validation results.
