# Stage 2C: Benchmark Quality Gates And Promoted Current Report

## Context

The current quality gate parser works, but Stage 2 needs clearer acceptance gates and a refreshed current benchmark report that references current commands and explicitly states remaining evidence gaps.

## Goal

Strengthen quality-gate reporting and regenerate/update the current benchmark report using current small benchmark runs.

## Ownership

You may edit:

- `benchmarks/quality_gate_report.py`
- `benchmarks/results/current_benchmark_report.md`
- `benchmarks/README.md`
- `README.md`
- tests for quality gate reporting

Do not edit SDK runtime or strategy implementations.

## In Scope

- Ensure quality gates include:
  - at least one non-random strategy has non-negative quality lift vs random;
  - win-rate is computable;
  - runtime summary is present;
  - manifest/evidence category is present when a manifest exists;
  - formula/native/external claims are not conflated.
- Run a small current benchmark smoke if needed and update the current report with actual command/results.
- Keep old Stage 9 numbers only as retained diagnostic evidence, not final proof.

## Out Of Scope

- Large benchmark reruns.
- Native external benchmark implementation; that is Stage 2B.

## Constraints

- Do not fabricate metrics. Only cite commands actually run.
- Keep tests fast.
- If a large benchmark is too expensive, document it as not run and keep a smoke evidence row.

## Suggested Validation

- `uv run pytest tests/test_quality_gate_report.py tests/test_project_smoke_benchmark.py tests/test_sdk_first_benchmark_real_datasets.py -q`
- `uv run python benchmarks/sdk_first_benchmark.py --preset smoke --output-dir benchmarks/results/stage2_smoke_current --overwrite`
- `uv run python benchmarks/quality_gate_report.py benchmarks/results/stage2_smoke_current`

## Acceptance Criteria

- Current benchmark report is internally consistent with actual current commands.
- Quality gate tests cover metadata-aware reports.
- Full suite remains green.
