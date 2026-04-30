# W51 - Fix Stage 0 README Dependency Wording

## Context
Final Stage 0 system review R66 found one valid documentation accuracy issue: README says core dependencies are pinned in `pyproject.toml`, but `pyproject.toml` contains at least one constrained range (`numpy>=2,<3`) while exact resolution is captured in `uv.lock`.

## Goal
Fix the README wording so dependency documentation accurately reflects the current project files.

## Responsibility Boundaries
- This is a tiny documentation-only fix.
- Keep the edit limited to the dependency wording around the Requirements section.

## In Scope
- `README.md`

## Out of Scope
- Do not edit dependencies.
- Do not edit `pyproject.toml` or `uv.lock`.
- Do not edit benchmarks, tests, source code, or other docs.

## Files That May Be Changed
- `README.md`

## Files That Must Not Be Touched
- All other files.

## Required Change
- Replace the inaccurate claim that core dependencies are "pinned in pyproject.toml" with wording that says dependencies are declared/constrained in `pyproject.toml` and the exact local resolution is captured by `uv.lock`.

## Validation
- Search README for stale wording around "pinned in pyproject.toml".
- Confirm no `42 passed` reappears.
- Do not run long tests; this is docs-only.

## Forbidden Actions
- Do not run destructive git commands.
- Do not change dependency versions.
- Do not broaden into Stage 1 implementation.

## Acceptance Criteria
- R66's P3 finding is addressed.
- The README is factually consistent with `pyproject.toml` and `uv.lock`.
