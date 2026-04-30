Task ID: W46
Short name: run reference benchmark analysis
Relation to overall task: Produce the actual SDK-vs-manual reference benchmark evidence after R59 accepted the harness.

Assumptions and resolved ambiguities:
- Reference harness is accepted.
- External libraries are not importable locally; optional external strategies should be skipped with explicit reasons.
- Formula-equivalence excludes random and should show exact parity if SDK formulas are correct.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Run the full reference benchmark over all diagnostic datasets, key SDK/manual strategies, and three seeds.
- Write a concise analysis file that answers whether SDK pure heuristics are weaker than manual formulas and whether current SDK improvements are justified.

Responsibility boundaries:
- In scope:
  - `benchmarks/results/reference_full/`
  - optional `benchmarks/results/reference_full/analysis.md`
- Out of scope:
  - `src/**`
  - `tests/**`
  - benchmark harness code
  - docs/README except later after review

Command:
`uv run python benchmarks/reference_strategy_benchmark.py --preset full --seeds 13,17,23 --output-dir benchmarks/results/reference_full`

Analysis requirements:
- Report row counts and skipped optional external libraries.
- Report formula-equivalence: rows, mean/min Jaccard, exact matches.
- Compare SDK pure strategies vs their manual equivalents on macro-F1 AULC and selected IDs.
- Compare best SDK improved strategy/mix against pure entropy/margin/least-confidence/manual baselines.
- State whether any SDK formula implementation appears worse than manual logic.
- State remaining limits: no direct modAL/skactiveml runtime because packages unavailable; BADGE/CoreSet not probability-only.

Expected validations:
- Run benchmark command.
- Strict JSON parse result JSON files.
- CSV row-count validation.
- Recompute headline aggregates for analysis.

Acceptance criteria:
- `benchmarks/results/reference_full/` contains metrics, selections, equivalence, summary, manifest, validation, external adapter status, and analysis.
- Analysis clearly answers the user's question with evidence.
