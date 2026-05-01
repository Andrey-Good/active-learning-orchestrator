# w93 - Audit Benchmark Comparison

## Context

The user requested benchmark evidence showing where this SDK is worse than analogs or manual implementations. Existing benchmark artifacts compare SDK strategies, but the audit needs a clear, reproducible comparison harness and an honest critique of benchmark validity.

## Goal

Review benchmark code and add a small audit benchmark that compares SDK overhead/quality against a direct manual baseline and any already-available reference adapters without adding new dependencies.

## Responsibility Boundaries

Own only:

- `benchmarks/audit_sdk_vs_manual.py`
- `tests/test_audit_benchmark_comparison.py`
- read-only inspection of `benchmarks/**`
- read-only inspection of `tests/test_reference_strategy_benchmark.py`
- read-only inspection of `docs/reference_active_learning_libraries.md`
- read-only inspection of `README.md`

Do not edit SDK implementation files. Do not edit existing benchmark outputs unless the new benchmark explicitly writes to a new audit path.

## In Scope

- Measure or expose SDK orchestration overhead versus a direct/manual loop on a tiny deterministic workload.
- Check whether existing benchmark claims are valid and whether they compare against true analogs.
- Create a testable benchmark script with stable output.
- Identify benchmark blind spots: dataset leakage, tiny samples, seed sensitivity, unfair baselines, artifact staleness.

## Out Of Scope

- Installing `modAL`, `skactiveml`, or other external libraries.
- Long dataset downloads.
- GPU workloads.
- Fixing benchmark implementation defects outside the new audit harness.

## Files/Areas Must Not Touch

- `src/**`
- existing benchmark result directories, unless writing to `benchmarks/results/audit_*`.
- runtime/strategy audit tests.

## Architectural Constraints

- The harness must be deterministic and runnable locally without network.
- Comparisons against unavailable external libraries must be reported as unsupported, not fabricated.
- Benchmark output must include environment-independent caveats.

## Special Attention

- The SDK may be slower than manual selection because of state, cache, and backend abstractions; quantify this on a small loop if feasible.
- Check whether benchmark metrics compare matching budgets and seeds.
- Check if quality gates can be gamed by random-equivalent or saturated runs.

## Forbidden Actions

- No dependency upgrades.
- No network downloads.
- No destructive git operations.
- Do not revert user changes.

## High-Level Plan

1. Inspect existing benchmark harness and docs.
2. Design a minimal manual-vs-SDK comparison that avoids external dependencies beyond project deps.
3. Add `benchmarks/audit_sdk_vs_manual.py`.
4. Add `tests/test_audit_benchmark_comparison.py` for deterministic structure and sanity.
5. Run the new test and, if fast, the benchmark itself.
6. Return findings and changed paths.

## Acceptance Criteria

- New benchmark harness can run from the repo root.
- New test verifies benchmark output shape and basic comparison semantics.
- The final response names where SDK is worse, where evidence is insufficient, and which analog comparisons were not run.

## Expected Tests And Validations

- `uv run pytest tests/test_audit_benchmark_comparison.py -q`
- `uv run python benchmarks/audit_sdk_vs_manual.py --output-dir benchmarks/results/audit_sdk_vs_manual_smoke`

## Dependencies

None.

## Parallel/Sequential Execution

Can run in parallel with runtime and strategy audit tasks. Write scope is disjoint.
