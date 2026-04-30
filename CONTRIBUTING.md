# Contributing

Thanks for considering a contribution.

This SDK is used to orchestrate active learning workflows, so correctness and reproducibility matter more than clever shortcuts.

## Development Setup

```bash
uv sync --dev
```

Run the standard checks before opening a pull request:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
uv build
uv run --with twine twine check dist/*.tar.gz dist/*.whl
```

## Contribution Rules

- Preserve public API compatibility unless the change is explicitly proposed as breaking.
- Do not change strategy selection order, tie-breaking, cache keys, split resolution, or benchmark evidence semantics without targeted regression tests.
- Do not add heavy dependencies to the core package. Use optional extras.
- Do not commit generated workdirs, `.agents/`, benchmark scratch output, Docker runtime state, credentials, or local caches.
- Add tests for new public behavior and edge cases.
- Keep benchmark claims scoped. Do not claim universal strategy superiority from small diagnostic evidence.

## Pull Request Checklist

- Tests pass locally.
- Ruff and mypy pass.
- README/docs are updated when public behavior changes.
- New config or product logic is documented in a spec, test, or docs page.
- Any benchmark evidence names exact dataset, budget, seed, cap, model, and strategy settings.

## Code Style

The project uses Ruff and Mypy. Prefer clear, boring code over clever abstractions. Internal refactors are welcome only when they are behavior-preserving and covered by tests.
