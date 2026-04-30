# W82 - Stage 9 Final Benchmark Report

## Context

The SDK now includes many more strategies, stop policies, reporting, simulator backend, managed Label Studio assets, and state safety than the stale README describes. Stage 9 needs fresh, honest benchmark evidence from the current codebase.

## Goal

Run the benchmark harness against the current SDK and update the final benchmark report artifacts with clear, reproducible headline metrics suitable for README inclusion.

## Responsibility Boundaries

You own benchmark execution and benchmark-result documentation only.

## In Scope

- `benchmarks/results/stage9_final/`
- `benchmarks/results/stage9_reference/`
- `benchmarks/results/current_benchmark_report.md`
- If needed, small benchmark harness/reporting fixes in:
  - `benchmarks/sdk_first_benchmark.py`
  - `benchmarks/reference_strategy_benchmark.py`
  - `benchmarks/README.md`

## Out of Scope

- README root update
- SDK strategy implementation changes
- Public API changes outside benchmark harness
- Docker/Label Studio code

## Must Not Touch

- `src/active_learning_sdk/**`
- `tests/**`
- `pyproject.toml`

## Required Benchmark Coverage

- SDK-first full or near-full run over all deterministic diagnostic datasets.
- Include random baseline and all currently implemented major strategy families:
  - uncertainty: random, entropy, margin, least_confidence
  - balanced/group: group_diverse_entropy, class_balanced_entropy, class_group_balanced_entropy
  - embedding/diversity: coreset_kcenter, embedding_kmeans_pp, max_min_embedding, density_weighted_diversity, deduplicate_near_neighbors
  - BADGE
  - stochastic/committee proxies where the harness supports them
  - hybrid/mix strategies where the harness supports them
- Include stop-policy diagnostics.
- Include reference/equivalence run for formula-equivalent uncertainty/balanced strategies.

## Execution Plan

1. Inspect benchmark CLI defaults and strategy availability.
2. Run a quick smoke if needed to verify the harness.
3. Run a final Stage 9 SDK benchmark into `benchmarks/results/stage9_final/`.
4. Run a final Stage 9 reference benchmark into `benchmarks/results/stage9_reference/`.
5. Parse generated CSV/JSON artifacts and update `benchmarks/results/current_benchmark_report.md` with:
   - command lines;
   - dataset/model/strategy coverage;
   - best strategy by macro-F1 AULC;
   - random baseline comparison;
   - small-budget comparisons;
   - group concentration diagnostics;
   - stop-policy savings/quality tradeoff;
   - reference/equivalence diagnostics;
   - limitations and what not to overclaim.
6. Keep all claims tied to generated artifacts.

## Acceptance Criteria

- Final benchmark commands complete successfully.
- Generated JSON is strict parseable JSON.
- `current_benchmark_report.md` is updated with current numbers, not stale Stage 0 numbers.
- Report is concise enough to be useful in README but detailed enough to audit.
- Any skipped strategy/library is explicitly explained.

## Expected Validation

- Re-run or parse generated artifacts after writing the report.
- Mention exact commands run and row counts in final response.

## Dependencies

Depends on completed Stage 8. Blocks root README benchmark section update.
