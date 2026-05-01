# w88 - Strengthen Quality Gate

## Context

The final review found that `benchmarks/quality_gate_report.py` can pass a strategy that merely ties random on a tiny run. That is too weak for product-quality validation.

## Goal

Strengthen the quality gate so it cannot pass on random-equivalent evidence alone. Keep the report useful for small diagnostic runs, but make the "PASS" signal require a positive quality signal.

## Responsibility Boundaries

Own only:

- `benchmarks/quality_gate_report.py`
- `tests/test_quality_gate_report.py`
- README benchmark wording only if required by the new gate semantics

Do not edit SDK strategy code or benchmark runner dataset construction.

## In Scope

- Require at least one non-random strategy to show positive mean final macro-F1 lift OR positive mean AULC lift, while still requiring no negative final/AULC lift and sufficient non-loss rate.
- Keep random baseline completeness checks strict.
- Add or update tests so a pure tie no longer passes.
- Add a test for a saturated-but-positive case if needed, so exact ties do not become the only supported saturated scenario.
- Keep JSON output strict and backward-readable.

## Out Of Scope

- No changes to benchmark execution.
- No changes to strategy implementations.
- No dependency changes.

## Constraints

- The quality gate should still pass existing meaningful benchmark artifacts once reports are regenerated.
- Use explicit named checks and clear details in generated reports.
- Do not introduce arbitrary high thresholds that would reject valid small-budget research runs.

## Acceptance Criteria

- `tests/test_quality_gate_report.py` passes.
- A random-equivalent strategy fails the quality gate.
- A strategy with non-negative final/AULC, non-loss >= 0.75, and at least one positive lift passes.
- Existing real/synthetic reports can be regenerated without code errors.

## Tests

- `uv run pytest tests/test_quality_gate_report.py -q`

## Dependencies

None.
