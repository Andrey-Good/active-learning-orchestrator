# R104 - Review Real Dataset Benchmark Fixes

## Context

R103 found two benchmark protocol issues:

- Initial seed selection for real many-class datasets was label-order biased when `initial_seed_size < label_count`.
- Zero-recall metrics counted labels with no test support.

Fixes were applied:

- `choose_initial_seed()` now shuffles label order before taking one sample per label.
- `train_and_evaluate()` computes zero-recall over labels present in the test split only.
- Tests were added/updated.
- A live `banking77` probe passed after switching Banking77 labels to `label_text`.

## Goal

Review the post-fix real-dataset benchmark layer.

## Responsibility Boundaries

You are reviewer only. Do not edit files.

## In Scope

- `benchmarks/sdk_first_benchmark.py`
- `tests/test_sdk_first_benchmark_real_datasets.py`
- `benchmarks/README.md`

## Out of Scope

- SDK runtime code under `src/**`
- Long full HF benchmark runs

## Special Attention

- Initial seed is now seed-dependent and not sorted-label-prefix biased.
- Synthetic class-covering behavior remains acceptable.
- Zero-recall excludes labels absent from test support.
- Banking77 uses `label_text`, not numeric label ids.
- Real presets default to `banking77` and `clinc_oos_imbalanced` as intended.
- Tests still avoid network.

## Validation Already Run

- `uv run pytest tests/test_sdk_first_benchmark_real_datasets.py tests/test_sdk_first_benchmark_embedding_diagnostics.py -q` -> `25 passed`.
- Live probe:
  `uv run python benchmarks/sdk_first_benchmark.py --preset real_smoke --datasets banking77 --strategies random,entropy --budgets 100 --initial-seed-size 16 --seeds 13 --output-dir benchmarks/results/runtime/real_smoke_probe`
  passed and produced strict JSON/CSV artifacts.

## Required Output

Findings first with severity and exact file/line references, or explicitly state no findings. Mention checks run.
