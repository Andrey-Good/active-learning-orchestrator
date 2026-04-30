# W08B Ingestion, State, Cache, Report, Export Stress

Task identifier: `wave8-w08b-ingestion-state-cache`

## Goal

Stress documented dataset ingestion and lifecycle behavior as an external SDK user: DataFrame, CSV, Parquet where dependencies are available, explicit splits, fingerprint mismatch, resume, cache stats, report generation, exports, locks, and corrupt/partial workdirs.

## Ownership

May write only under `.agents/tmp/blackbox_stress_wave8/w08b_state_cache/`.

Must not touch product files, docs, tests, benchmark source, or other workers' directories.

## In Scope

- Public SDK calls documented in README and `docs/SDK_CONTRACTS.md`.
- Optional `pandas`/Parquet dependencies if already available through `uv` or installable extras.
- Provider-style dataset and DataFrame/CSV/Parquet inputs.
- State reopen through `attach_runtime(...)` if exposed by public facade.
- `status`, `validate`, `list_rounds`, `get_round`, `import_labels`, `generate_report`, `export_labels`, `export_dataset_split`, `cache_stats`, `clear_cache`, `close`.
- Lock contention, partial workdir, corrupt state file, dataset mismatch, repeated open/close.

## Out Of Scope

- Reading implementation source under `src/**`.
- Reading repository tests under `tests/**`.
- Editing SDK code.
- Benchmark quality claims.

## Plan

1. Build a harness with isolated per-case workdirs under the owned tmp directory.
2. Run baseline simulator flow to prove setup.
3. Exercise each public lifecycle operation and validate generated artifacts externally.
4. Mutate external inputs between reopen attempts to test documented fingerprint protection.
5. Record JSON/CSV/Markdown/HTML/report/manifest observations and hashes when useful.
6. Write `findings.md` and `results.json`.

## Acceptance Criteria

- All confirmed issues have reproduction commands and artifact evidence.
- Non-issues and inconclusive optional dependency gaps are explicitly separated.
