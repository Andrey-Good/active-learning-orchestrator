# 2026-04-30 Release Readiness: Black-Box Package And Docs

## Task Identifier

release-readiness-blackbox-package-docs

## Context

Assess the SDK as a public user would see it, using documentation, built artifacts, package metadata, install/import behavior, examples, and command output. Follow the `project-black-box-stress-test` spirit: prefer observable behavior and reproducible evidence.

## Goal

Determine whether the packaged SDK and public-facing documentation are ready for a public GitHub release.

## Responsibility Boundaries

In scope:

- `README.md`, public docs, package metadata, `dist/` artifacts, install/import behavior, quickstart behavior, package contents.
- Commands such as `uv build`, `twine check`, package archive inspection, import smoke tests, README/example sanity checks.

Out of scope:

- Deep source-code implementation audit.
- Secret scanning beyond obvious docs/package content.
- Modifying files.

## Files Or Areas That May Be Read

- `README.md`
- `docs/`
- `pyproject.toml`
- `dist/`
- `benchmarks/README.md`
- built wheel/sdist metadata and contents

## Files Or Areas That Must Not Be Touched

- Do not edit any file.
- Do not inspect production source code under `src/` except by importing the installed package as an external user.

## Special Attention

- README encoding and bilingual section quality.
- Broken links or package URLs pointing to the wrong repo/name.
- Whether published artifacts include unexpected local reports, bulky benchmark data, or private/internal notes.
- Whether the quickstart actually works from a clean install.

## Acceptance Criteria

- List confirmed blockers/risks with evidence and reproduction commands.
- Note checks that passed.
- Clearly separate blocker, high, medium, and low severity items.

## Dependencies

None. Can run in parallel with the other release-readiness reviews.
