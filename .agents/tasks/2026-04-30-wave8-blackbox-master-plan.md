# Wave8 Black-Box Stress Master Plan

Task identifier: `wave8-blackbox-master-plan`

## Context

The user requested a black-box QA stress test using the project-black-box-stress-test skill. This project is an active-learning SDK with a documented public surface in `README.md`, `docs/README.md`, `docs/SDK_CONTRACTS.md`, and package metadata in `pyproject.toml`.

This campaign must not inspect implementation source code, repository tests, or benchmark implementation source. It may use public docs, public package metadata, public imports, documented CLI commands, runtime behavior, logs, generated artifacts, and prior black-box reports as historical context. Any prior finding must be rechecked against the current worktree before being reported as current.

## Broad Test Map

- Public facade import and configuration contract.
- Stable root exports and lazy optional adapter access.
- Dataset provider behavior with valid and invalid sample ids, missing samples, duplicate ids, changed payloads, explicit splits, and schema drift.
- Model adapter contracts: `fit`, `evaluate`, `predict_proba`, stochastic and committee probability shapes, embeddings, model id changes, and invalid return values.
- Public exception taxonomy: configuration, dataset mismatch, model adapter, backend, infrastructure, state corruption, lock conflicts, and stop criteria.
- Scheduler modes and strategy names documented in README: `single`, `mix`, `mix_interleaved`, `hybrid`, `custom`, `bandit`, and representative strategy families.
- Label backend behavior through simulator and custom injected backend objects; live Label Studio is optional only if a clean local service is available.
- State, resume, audit event, report, export, cache stats, cache clear, and dataset split behavior.
- DataFrame, CSV, and Parquet ingestion with optional extras, including payload/meta/group preservation and fingerprint mismatch on resume.
- Packaging and developer experience: build, twine check, isolated install, root import without optional dependencies, optional extras probes, README quickstart, documented benchmark commands.
- Benchmark quality and stress behavior: synthetic hostile data, capped documented benchmark presets, strategy regressions against matched random, duplicate/group concentration, low-budget failure modes, and warnings.
- Failure modes: malformed config, malformed model outputs, malformed backend returns, timeouts, repeated run/close/open, lock contention, partial/corrupt workdirs, and unsupported optional integrations.

## Decomposition

This can be split into independent black-box areas without overlapping write scopes:

- W08A owns public API, model adapter, custom backend, and exception taxonomy probes.
- W08B owns ingestion, state, resume, cache, report, export, and filesystem lifecycle probes.
- W08C owns packaging, docs examples, install/extras, and documented CLI smoke probes.
- W08D owns strategy and benchmark quality stress.

Parallel execution is allowed because workers write only under their assigned `.agents/tmp/blackbox_stress_wave8/<worker>/` directory and must not modify product files.

Reviewer sequence:

- R08A reviews W08A plus relevant W08B public-boundary findings.
- R08B reviews W08C plus W08D packaging/quality findings.
- A final orchestrator review merges only rechecked, well-supported findings.

## Global Rules

- Do not read files under `src/**`, `tests/**`, or benchmark implementation files such as `benchmarks/*.py`.
- Do not modify SDK source, tests, docs, package metadata, or benchmark source.
- New harnesses, logs, venvs, generated runs, and notes must stay under `.agents/tmp/blackbox_stress_wave8/`.
- Every confirmed issue must include reproduction, expected behavior, actual behavior, evidence, severity, confidence, and a suggested regression test.
- Reject weak findings caused by incorrect usage, missing optional dependencies outside the tested extra, unrealistic input, external service instability, or historical reports not reproduced on the current worktree.

## Acceptance Criteria

- Worker reports exist for each assigned area.
- Review reports challenge the strongest findings.
- Final report is saved under `.agents/tmp/blackbox_stress_wave8/final_report.md`.
- Final report distinguishes confirmed defects, limitations, residual risks, and rejected findings.
