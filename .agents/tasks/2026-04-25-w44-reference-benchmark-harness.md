Task ID: W44
Short name: reference benchmark harness
Relation to overall task: Build a reproducible comparison harness for SDK strategies against manual reference implementations and optional external-library adapters.

Goal and expected result:
- Add a benchmark script that can compare SDK selection behavior and learning curves to manual NumPy/scikit-learn reference implementations under identical model, pool, seed, and budget conditions.
- Include optional external-library hooks only if libraries are importable; benchmark must run without them.

Responsibility boundaries:
- In scope:
  - `benchmarks/reference_strategy_benchmark.py`
  - `benchmarks/README.md`
  - tests under `tests/` for manual reference parity helpers if practical
- Out of scope:
  - `src/**`
  - `pyproject.toml`
  - `uv.lock`
  - root README until results are reviewed
  - benchmark result artifacts (running the benchmark is a later task unless needed for smoke)

Important constraints:
- Reuse dataset/model helpers from `benchmarks/sdk_first_benchmark.py` where possible.
- Do not duplicate large code if importable functions/classes can be reused.
- Manual references must use the same `predict_proba` outputs as SDK `SelectionContext`.
- Track selected IDs, overlap/Jaccard with SDK equivalent, macro-F1 AULC, early macro-F1, rare recall, group HHI/top-group fraction.
- Make tie-breaking explicit and deterministic.
- External-library adapters must be optional and skipped with a clear reason if unavailable.

Suggested benchmark families:
- SDK: `random`, `entropy`, `margin`, `least_confidence`, `class_group_balanced_entropy`, `mix_interleaved_class_group_random`.
- Manual: `manual_entropy`, `manual_margin`, `manual_least_confidence`, `manual_class_group_balanced_entropy`, and deterministic `manual_random`.
- Optional external: `modal_entropy`, `modal_margin`, `modal_uncertainty`; `skactiveml_entropy/margin/least_confidence` if feasible.

Execution plan:
- Implement manual score functions and deterministic batch selectors.
- Implement a shared run loop using existing synthetic datasets and sklearn adapter.
- Add selection-equivalence diagnostics for equivalent strategies.
- Add CLI with presets: `smoke`, `full`.
- Write strict JSON/CSV/Markdown artifacts under a chosen output dir.
- Add focused tests for manual entropy/margin/least-confidence score ordering against simple probabilities.

Acceptance criteria:
- Smoke command runs quickly and writes artifacts.
- The script can run without external libraries.
- Equivalent SDK/manual selections are directly comparable.
- Tests pass.

Expected validations:
- `python -m py_compile benchmarks/reference_strategy_benchmark.py`
- focused pytest tests if added
- smoke benchmark command
