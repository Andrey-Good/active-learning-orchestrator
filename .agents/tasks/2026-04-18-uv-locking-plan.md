# UV Locking Plan

## Goal

Move the notebook environment from ad hoc `uv pip install` usage to normal project-based `uv` management with:
- `pyproject.toml`;
- `uv.lock`;
- synchronized environment via `uv sync`.

## Scope

- Create `pyproject.toml` with only the direct dependencies required by the notebook.
- Generate `uv.lock`.
- Sync the existing `.venv` using standard `uv` project commands.
- Re-run the notebook smoke test after the environment is locked.

## Constraints

- Do not modify unrelated repository files.
- Keep dependency scope minimal.
- Preserve notebook behavior that already passed smoke validation.

## Steps

1. Create a minimal non-packaged project definition in `pyproject.toml`.
2. List only direct notebook dependencies.
3. Generate `uv.lock` from the project definition.
4. Run `uv sync` against the project `.venv`.
5. Re-run the notebook smoke test.
6. Report the resulting files and status.

## Must Do

- Use normal `uv` workflow, not `uv pip`.
- Keep lockfile reproducible.
- Keep the dependency list readable and explicit.

## Must Not Do

- Do not widen the dependency list without need.
- Do not change SDK source files.
- Do not restructure the project.

## Validation

- `pyproject.toml` exists.
- `uv.lock` exists.
- `uv sync` completes successfully.
- Notebook smoke run still succeeds.
