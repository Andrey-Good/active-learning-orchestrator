# R103 - Review Real Dataset Benchmark Layer

## Context

W85 added opt-in Hugging Face real-dataset benchmark support for CLINC OOS, Banking77, and DAIR emotion, plus new class-coverage/budget metrics and tests.

## Goal

Review the real-dataset benchmark implementation for correctness, fairness, and test adequacy.

## Responsibility Boundaries

You are reviewer only. Do not edit files.

## In Scope

- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md`
- `tests/test_sdk_first_benchmark_real_datasets.py`
- Any directly relevant benchmark tests

## Out of Scope

- SDK runtime strategy code under `src/**`
- README root update unless a doc conflict is found
- Full expensive HF benchmark runs

## Special Attention

- Existing `smoke` and `full` presets must remain synthetic-only and offline.
- Real presets must be opt-in.
- Real many-class datasets must not require `initial_seed_size >= label_count`.
- Initial seed for real datasets must still have at least two classes for sklearn.
- Opaque sample/group IDs must not leak labels or raw numeric labels.
- Coverage/missing-class/zero-recall metrics must be correct and strict-JSON safe.
- GoEmotions must be deferred, not silently coerced to single-label.
- Tests should not require network.

## Forbidden Actions

- Do not edit files.
- Do not revert unrelated changes.

## Review Plan

1. Inspect implementation and tests.
2. Run targeted benchmark tests.
3. Run a tiny fake-loader CLI if useful.
4. Report findings first with severity and exact file/line references, or state no findings.

## Acceptance Criteria

- The real-dataset benchmark layer is fair, opt-in, and tested without network.
- Any residual risks are clearly stated.
