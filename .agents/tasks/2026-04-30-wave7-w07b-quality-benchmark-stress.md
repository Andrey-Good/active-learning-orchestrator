# W07B - Quality And Benchmark Stress

## Task Identifier

W07B-QUALITY-BENCHMARK-STRESS

## Context

Wave7 needs fresh evidence on whether documented strategies and benchmark claims survive hostile dataset/budget settings. Previous waves found and then closed several quality-gate issues; this pass should broaden coverage and avoid double-counting fixed historical defects.

## Goal

Run documented benchmark commands and bounded variants across synthetic and real datasets to expose low metrics, selection collapse, bad calibration, runtime outliers, or documentation/CLI mismatch.

## Responsibility Boundaries

May write only:
- `.agents/tmp/blackbox_stress_wave7/w07b_quality/**`

Must not touch:
- `src/active_learning_sdk/**`
- `tests/**`
- `benchmarks/*.py`
- existing benchmark result directories outside this task directory.

## In Scope

- Documented `benchmarks/sdk_first_benchmark.py` and `benchmarks/quality_gate_report.py` commands.
- Synthetic datasets, real capped datasets, multiple seeds, multiple budgets, random baseline comparisons.
- Metrics from generated CSV/JSON/Markdown artifacts.
- Runtime, non-loss, AULC, macro-F1, calibration, selection differentiation, budget warning behavior.

## Out Of Scope

- Reading benchmark implementation source.
- Fixing benchmark code.
- Native external-library benchmarks unless already documented and bounded.

## Architectural Constraints

Run benchmark scripts as documented black-box CLIs only. Do not inspect their source. Bound real-data runs so the task finishes and artifacts are reproducible.

## Special Attention

Metric complaints require matched random baseline evidence. If a run times out or is skipped due to missing optional deps/network, record it as environment/operational evidence, not automatically an SDK bug.

## Execution Plan

1. Read README, docs/BENCHMARK_EVIDENCE.md, and benchmarks/README.md only.
2. Run a synthetic hostile matrix.
3. Run at least one capped real dataset matrix beyond already-trivial smoke if time permits.
4. Run quality-gate report on completed directories.
5. Summarize metrics into `metric_summaries.json` and `findings.md`.

## Acceptance Criteria

- At least one synthetic and one real/capped benchmark attempt.
- Every completed benchmark output directory is named in the report.
- Findings distinguish hard failure, metric weakness, runtime outlier, and docs/claim mismatch.

## Dependencies

Can run in parallel with W07A/W07C/W07D.
