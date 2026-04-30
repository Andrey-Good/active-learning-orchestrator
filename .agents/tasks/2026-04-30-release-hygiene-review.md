# Task: 2026-04-30 Release Hygiene Review

## Context
Implementation changed repository hygiene only: package URLs, sdist allowlist, `.gitignore`, README/docs wording, benchmark README overwrite semantics, and one non-behavioral source comment.

## Goal
Review whether these changes are safe, behavior-neutral, and sufficient for the requested cleanup pass.

## Responsibility Boundaries
Read-only review of:
- `.gitignore`
- `pyproject.toml`
- `README.md`
- `docs/README.md`
- `benchmarks/README.md`
- `src/active_learning_sdk/engine.py`
- built `dist/active_learning_sdk-0.1.0*` artifact contents

## In Scope
- Identify broken links introduced by narrowing sdist.
- Identify public metadata problems.
- Identify accidental exclusion of required runtime package files.
- Check whether any code behavior changed beyond a comment.

## Out of Scope
- Editing files.
- Re-running the full benchmark matrix.
- Reworking algorithm implementations.

## Acceptance Criteria
- Say whether changes are safe to keep.
- List blockers if any.
- Include validation commands or artifact evidence.
