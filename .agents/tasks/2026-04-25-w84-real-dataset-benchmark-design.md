# W84 - Real Dataset Benchmark Design

## Context

The SDK benchmark suite currently uses deterministic synthetic datasets. The user wants real Hugging Face datasets added for quality measurement under active-learning budget constraints. We already identified strong candidates:

- `clinc/clinc_oos`, config `imbalanced`: 10,625 train / 5,500 test / 151 classes, random misses many classes at small budgets.
- `clinc/clinc_oos`, config `plus`: 15,250 train / 5,500 test / 151 classes, balanced-ish but many classes.
- `mteb/banking77`, config `default`: 9,993 train / 3,076 test / 77 classes.
- `google-research-datasets/go_emotions`, config `simplified`: 43,410 train / 5,427 test, multi-label; useful later, but not ideal for current single-label SDK benchmark without explicit conversion policy.
- `dair-ai/emotion`, config `split`: 16,000 train / 2,000 test / 6 classes; useful sanity dataset, but random coverage is too easy.

## Goal

Design a practical real-dataset benchmark plan and identify what should/should not be tested with each dataset.

## Responsibility Boundaries

You are an explorer/designer. Do not edit files unless explicitly asked later.

## In Scope

- `benchmarks/sdk_first_benchmark.py`
- `benchmarks/README.md`
- Tests under `tests/`
- Hugging Face dataset candidates and benchmark protocol.

## Out of Scope

- Implementing code.
- Running full expensive benchmarks.
- SDK algorithm changes.

## Questions To Answer

1. Which datasets should be included immediately in the repo benchmark harness?
2. Which datasets should be listed as not suitable for current single-label SDK metrics, and why?
3. What metrics are essential for budgeted active learning beyond accuracy/macro-F1?
4. What tests should be added to ensure the benchmark protocol is fair?
5. What presets should exist so local runs stay under practical time limits?

## Required Output

Return a concise implementation-ready design:

- recommended dataset registry entries;
- budget presets;
- metric additions;
- test list;
- limitations;
- acceptance criteria.

## Constraints

- Keep train-time per dataset/model strategy practical.
- Do not create label leakage via sample ids, group ids, metadata, or sorted order.
- Keep real-dataset download optional and not required for unit test suite.
