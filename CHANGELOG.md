# Changelog

All notable changes to this project are documented here.

The project currently follows a pre-1.0 beta versioning model. Public contracts are documented in [docs/SDK_CONTRACTS.md](docs/SDK_CONTRACTS.md).

## 0.1.0

Initial beta release candidate.

### Added

- Stateful `ActiveLearningProject` facade.
- Persistent project state, project locking, resume-safe rounds, and audit artifacts.
- Dataset fingerprinting and split identity validation.
- Simulator backend for deterministic tests.
- Label Studio external and managed Docker backend support.
- Uncertainty, class/group-balanced, embedding/diversity, BADGE, stochastic, committee, hybrid, mix, and experimental bandit strategy surfaces.
- Label import, status, validation, round inspection, label export, dataset split export, report generation, and cache management APIs.
- SDK-first, reference, native external, quality-gate, synthetic, and capped real-data benchmark harnesses.
- Optional sklearn, Hugging Face, datasets, xxhash, and benchmark extras.

### Validation

- `uv run pytest -q` -> `623 passed, 1 skipped`
- `uv run mypy src` -> `Success: no issues found in 38 source files`
- `uv run ruff check .` -> `All checks passed!`
- `uv build` -> successful wheel and source distribution
- `twine check` -> successful package metadata validation
