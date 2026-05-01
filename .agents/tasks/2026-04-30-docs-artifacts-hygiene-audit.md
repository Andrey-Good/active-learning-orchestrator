# Task: 2026-04-30 Docs And Artifacts Hygiene Audit

## Context
The repo has many generated audit, benchmark, notebook, and task artifacts. The user wants professional repository presentation without breaking functionality or losing important evidence.

## Goal
Classify docs/artifacts into publishable docs, internal evidence, generated outputs, and removable stale files.

## Responsibility Boundaries
Own read-only review of `README.md`, `docs/**`, `benchmarks/**`, `docker/**`, `lab/**`, `.agents/**`, and root generated files.

## In Scope
- README stale evidence and inaccurate links/counters.
- Internal audit reports that should not ship in sdist/public release.
- Generated benchmark outputs and caches.
- Deleted notebooks/lab files currently visible in git status.
- Documentation that should remain because it defines public contracts.

## Out of Scope
- Editing files.
- Rewriting public documentation.
- Deleting files.

## Must Not Touch
- Do not modify any file.
- Do not remove evidence directories.
- Do not collapse docs without identifying a safe destination.

## Acceptance Criteria
- Provide a publish/ignore/remove recommendation per artifact category.
- Flag any README claims that are stale or unsupported.
- Identify any docs that should remain in sdist.
- Include validation commands after fixes.
