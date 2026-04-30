# Stage 11D: Real-Data Benchmark Release Fixes

## Task Identifier

stage11d-benchmark-release-fixes

## Context

Stage 11A/11B/11C audits rejected benchmark release readiness. The shared
blockers are missing calibration metrics, weak real-report contracts, one-seed
standard evidence, and docs/quality-gate gaps.

## Goal

Implement the Stage 11 benchmark evidence contract without changing SDK runtime
behavior.

## Responsibility Boundaries

In scope:

- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/quality_gate_report.py`
- benchmark-related tests
- `benchmarks/README.md`
- `docs/BENCHMARK_EVIDENCE.md`
- `README.md` benchmark wording if needed

Out of scope:

- Production SDK runtime code under `src/active_learning_sdk/**`
- Long real-data benchmark execution
- Retained historical benchmark artifacts, except generated temporary test
  outputs under pytest tmp dirs
- Label Studio/backend changes

## Files May Be Changed

- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/quality_gate_report.py`
- `benchmarks/README.md`
- `docs/BENCHMARK_EVIDENCE.md`
- `README.md`
- `tests/test_sdk_first_benchmark_real_datasets.py`
- `tests/test_quality_gate_report.py`
- `tests/test_benchmark_evidence_contract.py`
- new benchmark contract tests if useful

## Required Fixes

1. Add calibration metrics to SDK-first benchmark outputs:
   - multiclass Brier score;
   - NLL/log-loss;
   - ECE.
   Compute them after each fit from test-set `predict_proba` aligned to
   `dataset.labels`. Include them in budget metrics and full-train reference
   rows.
2. Extend `quality_gate_report.py`:
   - parse and summarize calibration metrics;
   - render calibration summaries in Markdown;
   - fail Stage 11 real/promoted reports when required metrics are absent;
   - expose report-level seed counts and distinguish smoke vs standard evidence;
   - avoid classifying native external-only rows as formula comparison.
3. Define and enforce a standard real-report contract:
   - `real_medium` and `real_full` require explicit finite `--max-train-samples`
     and `--max-test-samples` unless an explicit local/uncapped override is added;
   - standard real evidence requires at least three seeds;
   - keep quick `real_smoke` usable for local probes, but label it smoke-only and
     do not let it masquerade as standard evidence.
4. Add tests:
   - calibration columns are written by a tiny SDK-first run;
   - calibration math is correct for a simple fixture;
   - quality gate includes calibration in JSON/Markdown;
   - real standard presets reject missing caps and too few seeds;
   - quality gate fails capped-real/standard evidence missing calibration/coverage
     metrics;
   - native external-only evidence category is not reported as formula comparison.
5. Update docs:
   - standard real commands show caps and `--seeds 13,21,34`;
   - calibration definitions and artifact columns are documented;
   - native external command support is separated from retained artifact evidence;
   - historical Banking77 two-seed evidence remains labeled retained/diagnostic,
     not Stage 11 standard.

## Architectural Constraints

- Do not invent benchmark numbers.
- Do not download datasets.
- Keep synthetic smoke cheap.
- Keep generated JSON strict (`allow_nan=False`).
- Missing calibration in legacy retained artifacts should be explainable; do not
  retroactively mutate old result files.

## Expected Validations

Run:

- `uv run pytest tests/test_quality_gate_report.py tests/test_sdk_first_benchmark_real_datasets.py tests/test_benchmark_evidence_contract.py -q`
- focused benchmark smoke test(s) added by this task
- `uv run pytest -q`
- `uv run mypy src`
- `uv run --with ruff ruff check .`

## Acceptance Criteria

- Stage 11A/11B/11C P1/P2 blockers are closed.
- No production SDK runtime files are changed.
- Full suite/static checks pass.
- Docs and tests prevent future evidence drift.

## Dependencies

- Stage 11A/11B/11C audit reports.

## Parallelism

Single worker task because benchmark runner, quality gate, and docs are coupled.
Review must be done by a separate reviewer.
