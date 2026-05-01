Task ID: W43
Short name: fix README test links
Relation to overall task: Resolve R53 documentation review findings before final system review.

Assumptions and resolved ambiguities:
- R53 found README references missing `tests/test_label_backends.py` and `tests/test_core_sdk.py`.
- Current test files are split differently.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Update README test/simulator references so all local test links point to existing files.
- Keep documentation accurate and concise.

Responsibility boundaries:
- In scope:
  - `README.md`
- Out of scope:
  - `benchmarks/**`
  - `src/**`
  - `tests/**`
  - pyproject/lockfiles

Architectural constraints and forbidden actions:
- Do not create test files just to satisfy links.
- Do not change benchmark numbers.
- Do not claim coverage that tests do not provide.

High-level execution plan:
- List current `tests/*.py`.
- Replace stale links with existing files and/or generic test-directory language.
- Search README for stale test filenames.

Acceptance criteria:
- README no longer references missing test files.
- All README local test links exist.

Expected validations:
- `Get-ChildItem tests -Filter *.py`
- Search README for `test_core_sdk.py` and `test_label_backends.py`.
- Optional quick local link existence check for Markdown links in README.

Dependencies:
- Fixes R53.

Parallel/sequential notes:
- Documentation-only task. Must be reviewed after completion.
