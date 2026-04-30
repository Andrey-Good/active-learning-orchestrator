# BB02 State, Artifacts, Cache, Resume Worker

## Goal

Black-box stress test persisted state, resume/reattach behavior, project locking, report/export APIs, cache observability, dataset mismatch protection, and state-corruption handling.

## Allowed Context

- Public docs: `README.md`, `docs/README.md`, `docs/SDK_CONTRACTS.md`.
- Package metadata: `pyproject.toml`.
- Runtime behavior through `uv run python`, generated workdirs, logs, and artifacts.

## Write Scope

- `.agents/tmp/blackbox/bb02-state-artifacts/**`

## Must Not Touch

- `src/**`
- `tests/**`
- Source-like implementation files.
- Existing project data outside the write scope.

## In Scope

- Configure/run/reopen project workdirs through public APIs.
- Exercise `attach_runtime(...)`, `status`, `validate`, `list_rounds`, `get_round`.
- Exercise `generate_report`, `export_labels`, `export_dataset_split`.
- Check cache stats/clear semantics from public docs.
- Simulate concurrent access through separate processes.
- Corrupt generated state files in the worker-owned workdir and observe public error behavior.
- Reopen with changed datasets/splits and observe mismatch handling.

## Out Of Scope

- Strategy quality comparisons.
- Benchmark harness and optional external integrations.
- Source-level state format inspection beyond generated files needed as black-box artifacts.

## Acceptance Criteria

- Produce `.agents/tmp/blackbox/bb02-state-artifacts/report.md`.
- Include minimal reproduction scripts/commands for confirmed issues.
- Preserve generated logs and relevant generated artifacts under the write scope.

## Expected Validation

- Use only worker-owned generated workdirs.
- Treat generated state/report files as external artifacts, not source.
