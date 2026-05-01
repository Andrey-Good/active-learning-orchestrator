# 2026-04-30 Black-Box Stress Test Map

## Task Identifier

BB-MAP: Active Learning SDK black-box QA stress test

## Context

Run a broad black-box QA stress test of the project using only public documentation, public examples, package metadata, runtime behavior, command output, generated artifacts, and logs. Source code inspection is forbidden for this effort.

## Primary Public Surfaces To Test

- Root package import and stable exports listed in `docs/SDK_CONTRACTS.md`.
- README simulator quickstart and public `ActiveLearningProject` lifecycle.
- Runtime object contract: dataset provider, model adapter, label schema, label backend, scheduler, annotation policy, split config.
- Model capability validation for probability, stochastic, committee, embedding, logits, and gradient-related strategies.
- Scheduler modes and strategy names documented in README.
- State persistence, reopen/resume, `attach_runtime(...)`, locking, dataset fingerprint mismatch protection, and state corruption handling.
- Public facade operations: `status`, `validate`, `list_rounds`, `get_round`, `import_labels`, `generate_report`, `export_labels`, `export_dataset_split`, `cache_stats`, `clear_cache`, `close`.
- Simulator backend behavior and idempotency from an external user viewpoint.
- Label Studio external and managed Docker configuration behavior, without relying on private implementation.
- Cache behavior and cache observability promises from the public contracts.
- Benchmark commands documented in `README.md` and `benchmarks/README.md`, including smoke/project smoke behavior and artifact schema promises.
- Packaging behavior: editable install, root import without optional extras, optional adapter access, build/twine metadata, Python version constraints.
- Documentation consistency and developer-experience hazards.

## Edge Cases And Failure Modes

- Empty datasets, duplicate sample ids, missing sample ids, missing text fields, non-string text, unstable sample ordering.
- Invalid labels, empty label schema, single-class schema where probability methods imply multiple columns, labels not in schema.
- Probability rows with wrong width, negative values, non-finite values, non-normalized values, wrong number of rows.
- Model adapters missing `fit`, `evaluate`, or required probability/embedding/stochastic/committee methods for selected strategies.
- Strategy batch sizes larger than pool, zero/negative batch sizes, exhausted pools, duplicate selections, selected ids not in pool.
- Resume with changed dataset content, changed sample ids, changed split assignments, changed runtime objects, or corrupted state files.
- Concurrent project open/run attempts against the same workdir.
- Report/export output into missing directories, existing files, unusual paths, and formats not documented.
- Cache reuse after model id changes or after model training without a changed model id.
- Benchmark command failures, non-empty output-dir handling, missing optional extras, real-dataset caps and seed requirements.
- Managed Docker credential absence and custom compose path misconfiguration.

## Decomposition

Parallelizable:

- BB01 Core public API and adapter/strategy contract behavior.
- BB02 Persistence, reports, exports, cache, resume, and concurrency behavior.
- BB03 Packaging, optional integrations, benchmarks, Label Studio config, and documentation/DX behavior.

Sequential:

- Worker execution must precede reviewer validation.
- Reviewer findings must be triaged before final report.
- Final system-level review must run after all accepted worker/reviewer findings are consolidated.

## Final Deliverables

- Raw evidence under `.agents/tmp/blackbox/`.
- Final repair-oriented report under `.agents/tmp/blackbox/BLACK_BOX_STRESS_TEST_REPORT_2026-04-30.md`.
- Report entries must include reproduction path, expected behavior, actual behavior, evidence, severity, confidence, and regression-test recommendation.

## Forbidden Actions

- Do not inspect files under `src/` or source-code-like implementation files for product behavior.
- Do not read test source files as implementation guidance.
- Do not report issues without reproducing or directly observing the evidence.
- Do not modify project source code.
- Do not delete existing project artifacts.
