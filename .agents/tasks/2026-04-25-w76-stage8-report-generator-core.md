# W76 - Stage 8 Report Generator Core

## Context

Stage 8 is reporting and auditability. Current `ReportGenerator.generate_html()` is a scaffold that raises `NotImplementedError`, while the SDK now has enough state to produce useful product reports:

- project status/counts;
- dataset/config fingerprints;
- rounds, selected ids, scheduler snapshots, timeout traces;
- metrics history and learning curves;
- label distribution and needs-review counts;
- stop criteria traces;
- backend timeout/retry diagnostics where present.

## Goal

Implement a dependency-light report generator that emits strict JSON, Markdown, and HTML artifacts from `ProjectState`.

## Responsibility Boundaries

You may change:

- `src/active_learning_sdk/report.py`
- `src/active_learning_sdk/engine.py` only for report API plumbing
- `src/active_learning_sdk/project.py` only for report API plumbing
- `src/active_learning_sdk/__init__.py` only if needed for public export
- tests, preferably `tests/test_report_generation.py`

Do not edit benchmark harness, backend code, or README in this subtask.

## In Scope

- Replace scaffold report implementation with real functions.
- Keep dependencies optional-free: use stdlib only.
- Generate at least:
  - `summary.json` or user-specified JSON path with strict JSON (`allow_nan=False`);
  - Markdown report;
  - HTML report using escaped content and simple tables.
- Include:
  - run/project metadata: project name, state version, created/updated timestamps;
  - dataset fingerprint/source/schema;
  - counts by sample status;
  - label distribution from `sample_labels`;
  - metrics history;
  - round timeline including status, selected count, task count, resolved count, metrics, reward;
  - scheduler/acquisition snapshots;
  - stop trace from `scheduler_state["stop_trace"]`;
  - annotation timeout traces from round scheduler snapshots.
- Sanitize non-finite floats and non-JSON-serializable values.
- Make `ActiveLearningProject.generate_report(...)` usable without optional dependencies.
- Add tests for:
  - strict JSON report with NaN metrics sanitized;
  - Markdown/HTML files created and contain key sections;
  - timeout/stop traces are included;
  - public project method writes report artifacts.

## Out Of Scope

- No charts with matplotlib.
- No PDF.
- No README release docs yet.
- No benchmark result aggregation yet.

## Architectural Constraints

- Do not mutate project state while generating reports.
- Report output must be deterministic except timestamps already in state.
- Avoid embedding raw unsafe HTML without escaping.

## Acceptance Criteria

- `project.generate_report(...)` no longer raises for configured projects.
- Reports are useful enough to audit a run without opening `state.json`.
- Targeted tests and full suite pass.

## Validation

- `uv run --group dev pytest -q tests/test_report_generation.py`
- `uv run --group dev pytest -q`
