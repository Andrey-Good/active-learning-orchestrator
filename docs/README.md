# SDK Documentation

This directory contains public SDK documentation that is safe to ship in source distributions.

## Public Contract Documents

- `SDK_CONTRACTS.md` defines the stable, provisional, and internal SDK contract surfaces.
- `BENCHMARK_EVIDENCE.md` defines how benchmark artifacts may be cited as product evidence.
- `LABEL_STUDIO_LIVE_TESTS.md` documents opt-in live Label Studio tests.

## Public Entry Points

- The repository root `README.md` is the public install and quickstart entrypoint.
- The first-run quickstart uses the core package plus the simulator backend; it intentionally does not require pandas, optional extras, or a live Label Studio service.
- Human-labeling flows are documented separately as external Label Studio and managed Docker Label Studio paths.
- Current optional extras are `sklearn`, `huggingface`, `datasets`, `xxhash`, `benchmarks`, and `all`; feature mapping lives in the root README and package metadata in `pyproject.toml`.

## Evidence Policy

Benchmark evidence should name the exact artifact directory and whether it was generated with the current manifest/probability-validation schema. Legacy Stage 9 retained artifacts remain useful diagnostic evidence, but their manifests are pre-schema and do not prove current manifest metadata fields unless regenerated.

Internal audit notes, task documents, and black-box review artifacts may exist in the working tree, but they are not part of the public source distribution.
