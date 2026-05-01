# R95 - Review W76 Report Generator Core

## Context

W76 replaced the report scaffold with dependency-free JSON/Markdown/HTML report generation and added tests.

## Goal

Review report generation for audit usefulness, strict JSON safety, non-mutation of state, HTML escaping, and public API behavior.

## Responsibility Boundaries

You are a reviewer. Do not edit files.

## In Scope

- `src/active_learning_sdk/report.py`
- `src/active_learning_sdk/engine.py` report API changes
- `tests/test_report_generation.py`

## Required Review Questions

- Does `generate_report(...)` create useful JSON/Markdown/HTML artifacts?
- Are non-finite floats and non-serializable values sanitized for strict JSON?
- Does report generation avoid mutating project state?
- Is HTML escaped for user-controlled labels/sample ids/scheduler payloads?
- Are stop traces and annotation timeout traces included?
- Are tests meaningful and not overfitted?

## Validation To Run

- `uv run --group dev pytest -q tests/test_report_generation.py`
- `uv run --group dev pytest -q`

## Output

Return severity-ordered findings with file/line refs. If no findings remain, say so explicitly and include validation results.
