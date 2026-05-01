Task ID: R59
Short name: review reference random equivalence fix
Relation to overall task: Verify W45 resolved R58 false random equivalence before running the full reference benchmark.

Assumptions and resolved ambiguities:
- W45 claims formula-equivalence now excludes random and smoke parity is perfect for entropy/margin/least-confidence/class-group.
- Reviewer is read-only and must not edit files.

Goal and expected result:
- Confirm random is no longer included in formula-equivalence aggregate.
- Confirm docs/tests reflect random as a separate baseline rather than formula parity.
- Confirm smoke equivalence metrics are meaningful.
- Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvements.

Responsibility boundaries:
- Review scope:
  - `benchmarks/reference_strategy_benchmark.py`
  - `benchmarks/README.md`
  - `tests/test_reference_strategy_benchmark.py`
- Out of scope:
  - editing files
  - SDK source changes
  - dependency changes

Expected validations:
- `python -m py_compile benchmarks/reference_strategy_benchmark.py`
- `uv run --group dev pytest tests/test_reference_strategy_benchmark.py -q`
- Smoke command if cheap.

Acceptance criteria:
- If satisfied, explicitly say there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide concrete findings with severity and evidence.
