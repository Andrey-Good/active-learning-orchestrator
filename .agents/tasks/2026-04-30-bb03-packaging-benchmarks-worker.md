# BB03 Packaging, Benchmarks, Optional Integrations Worker

## Goal

Black-box stress test packaging/developer experience, optional extras behavior, documented benchmark commands, Label Studio configuration paths, and documentation consistency.

## Allowed Context

- Public docs: `README.md`, `docs/README.md`, `docs/BENCHMARK_EVIDENCE.md`, `docs/LABEL_STUDIO_LIVE_TESTS.md`, `benchmarks/README.md`, `docker/label_studio/README.md`.
- Package metadata: `pyproject.toml`.
- Runtime behavior through public commands, build/install commands, benchmark commands, generated artifacts, and logs.

## Write Scope

- `.agents/tmp/blackbox/bb03-packaging-benchmarks/**`

## Must Not Touch

- `src/**`
- `tests/**`
- Existing benchmark result directories except by running documented commands with a worker-owned `--output-dir`.
- Source-like implementation files outside public benchmark command execution.

## In Scope

- Validate documented benchmark smoke/project-smoke commands with worker-owned output directories.
- Validate benchmark refusal behavior for non-empty output dirs.
- Check manifest/artifact promises from docs.
- Check build metadata and install/root import behavior.
- Check optional adapter behavior with missing extras where feasible.
- Probe Label Studio managed Docker credential/config errors without requiring a live service.
- Identify documentation contradictions that are proven through runtime behavior.

## Out Of Scope

- Deep strategy correctness beyond benchmark-level black-box artifacts.
- Live Label Studio API testing unless the local environment is already configured and safe.

## Acceptance Criteria

- Produce `.agents/tmp/blackbox/bb03-packaging-benchmarks/report.md`.
- Include command transcripts or summarized logs for all confirmed/rejected findings.
- Do not create promoted benchmark artifacts outside the write scope.
