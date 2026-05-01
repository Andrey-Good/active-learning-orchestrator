# 2026-04-27-w111 Exhaustive benchmark/docs/packaging objections

## Context

The user wants all practical objections so the project can be cleaned in one batch. Prior audit found benchmark evidence limitations, stale docs, and microbenchmark overhead. This subtask should audit all docs, benchmark, packaging, generated artifact hygiene, and release-readiness issues.

## Goal

Enumerate objections covering README accuracy, stale docs, benchmark validity, external-library claims, generated files, packaging contents, optional dependencies, CI/lint/type-check gaps, and repo hygiene.

## Ownership

Read all files. You may run static tools via ephemeral `uv run --with ...` commands. Do not edit implementation or dependency files. Current dirty worktree is user-owned.

## In Scope

- `README.md`, `docs/`, `benchmarks/`, `pyproject.toml`, `.gitignore`
- package build outputs and sdist/wheel contents
- optional dependency declarations

## Out Of Scope

- SDK implementation fixes
- Adding long-running real dataset benchmarks

## Required Output

Return a categorized list of objections with severity, exact file references where possible, and commands run. Separate "must fix before release" from "cleanup before maintainability signoff".
