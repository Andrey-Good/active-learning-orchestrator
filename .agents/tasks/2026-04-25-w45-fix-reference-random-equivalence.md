Task ID: W45
Short name: fix reference random equivalence
Relation to overall task: Resolve R58 finding so SDK/manual equivalence diagnostics are not polluted by intentionally different random tie/randomization semantics.

Assumptions and resolved ambiguities:
- R58 found `random` vs `manual_random` is a false equivalence because the manual selector intentionally uses a different hash/seed contract.
- Formula strategies (`entropy`, `margin`, `least_confidence`, `class_group_balanced_entropy`) matched exactly in smoke.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Update reference benchmark equivalence diagnostics so aggregate SDK/manual parity only covers formula-equivalent strategies.
- Keep `manual_random` as a separate baseline if useful, but do not count it as formula equivalence with SDK random unless the selector is changed to exactly mirror SDK random.
- Update tests/docs accordingly.

Responsibility boundaries:
- In scope:
  - `benchmarks/reference_strategy_benchmark.py`
  - `benchmarks/README.md`
  - `tests/test_reference_strategy_benchmark.py`
- Out of scope:
  - `src/**`
  - pyproject/lockfiles
  - root README
  - benchmark result artifacts except temporary smoke outputs if needed

Architectural constraints and forbidden actions:
- Do not change SDK `RandomStrategy`.
- Do not hide random baseline; explain it as a stochastic/hash baseline with different deterministic contract.
- Keep strict JSON/CSV artifact behavior.

Execution plan:
- Remove `random: manual_random` from `EQUIVALENT_STRATEGIES`, or split equivalence categories so formula-equivalence excludes random.
- Update summary/report wording to say formula equivalence.
- Add/update a test that asserts random is not included in formula equivalence.
- Run focused tests and smoke benchmark.

Acceptance criteria:
- Reference smoke aggregate equivalence no longer reports random false mismatches.
- Formula-equivalent strategies show exact parity in smoke, or any mismatch is clearly reported.
- Tests pass.

Expected validations:
- `python -m py_compile benchmarks/reference_strategy_benchmark.py`
- `uv run --group dev pytest tests/test_reference_strategy_benchmark.py -q`
- `uv run python benchmarks/reference_strategy_benchmark.py --preset smoke --output-dir <temp>`

Dependencies:
- Fixes R58.
