# Task: wave3-load-cache-packaging

## Context
The user requested repeat black-box stress testing after fixes. Prior wave found Windows persistent-cache failures and cache observability inconsistency, while packaging passed. SDK source inspection remains forbidden.

## Goal
Re-test cache/load/package behavior and search for new defects under larger public workflows.

## Responsibility Boundaries
Owns only artifacts under `.agents/tmp/blackbox_stress/wave3_load/` and findings in `.agents/tmp/blackbox_stress/wave3_load_findings.md`.

## In Scope
- Read public docs and prior reports.
- Re-test persistent cache on Windows with entropy/probability strategies at 1,000+ samples.
- Re-test `cache_stats()` observability and `clear_cache()`.
- Run larger report/export/resume/reopen loops.
- Re-test packaging quick checks if local artifact behavior changed.
- Search for new load failures: stale temp files, repeated report generation, large IDs, timeout policies, many rounds, concurrent status/cache reads.

## Out of Scope
- Reading SDK source.
- Modifying SDK source, docs, tests, benchmarks, or sibling wave3 directories.
- Fixing defects.

## Must Not Touch
- `src/**`
- `tests/**`
- benchmark source files
- other wave3 worker directories

## Execution Plan
1. Build a fresh black-box load/cache harness under owned directory.
2. First classify old cache/load findings as fixed/still-open/changed.
3. Run additional pressure cases.
4. Optionally build/install package from local artifact if useful for packaging regression.

## Acceptance Criteria
- At least 15 load/cache/package cases attempted.
- At least two cases use >=1,000 samples.
- Explicit old-finding regression table.
- No source inspection.
