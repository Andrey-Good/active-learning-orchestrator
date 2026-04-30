Task ID: R58
Short name: review reference benchmark harness
Relation to overall task: Independent review of W44 reference benchmark before using its mismatch results to diagnose SDK quality.

Assumptions and resolved ambiguities:
- W44 added `benchmarks/reference_strategy_benchmark.py`, docs, and tests.
- Smoke run produced non-perfect SDK/manual exact matches, so harness correctness matters before interpreting results.
- Reviewer is read-only and must not edit files.

Goal and expected result:
- Verify the reference benchmark compares SDK and manual implementations fairly.
- Check manual formulas, shared model/probability use, equivalence diagnostics, optional external skips, and artifact validity.
- Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvements.

Responsibility boundaries:
- Review scope:
  - `benchmarks/reference_strategy_benchmark.py`
  - `benchmarks/README.md`
  - `tests/test_reference_strategy_benchmark.py`
  - temporary smoke artifacts only if available
- Out of scope:
  - editing files
  - SDK source changes
  - dependency changes

Review checks:
- Manual entropy/margin/least-confidence formulas match primary-source formulas.
- SDK and manual selectors share the same trained model, pool, initial seed, and budgets.
- Equivalent strategy pairs are reported clearly.
- Mismatches can be attributed to tie-breaking or actual score/selection differences.
- External-library absence is skipped, not hidden as success.
- Tests are meaningful.

Expected validations:
- `python -m py_compile benchmarks/reference_strategy_benchmark.py`
- `uv run --group dev pytest tests/test_reference_strategy_benchmark.py -q`
- Run smoke if cheap, or inspect current smoke output if available.

Acceptance criteria:
- If satisfied, explicitly say there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide concrete findings with severity and evidence.
