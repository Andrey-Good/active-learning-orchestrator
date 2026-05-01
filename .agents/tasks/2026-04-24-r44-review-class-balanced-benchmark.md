Task ID: R44
Short name: review class-balanced benchmark
Relation to overall task: Independent review of W34 benchmark results before using them to choose the next SDK algorithm improvement.

Assumptions and resolved ambiguities:
- W34 claims benchmark artifacts show `class_balanced_entropy` wins grouped duplicates and separable topics on macro-F1 AULC, but is mixed on rare-class macro-F1.
- Reviewer is read-only and must not edit files.

Goal and expected result:
- Verify the benchmark spec, artifacts, row counts, strict JSON validity, and `analysis.md` headline numbers.
- Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvements.

Responsibility boundaries:
- Review scope:
  - `benchmarks/sdk_first_benchmark.py`
  - `benchmarks/README.md`
  - `benchmarks/results/class_balanced_entropy/**`
- Out of scope:
  - `src/**`
  - `tests/**`
  - root README
  - Docker/backends

Review checks:
- `class_balanced_entropy` benchmark spec uses `SchedulerConfig(mode="single", strategy="class_balanced_entropy")`.
- Validation/leakage checks are not weakened.
- JSON artifacts parse strictly.
- `metrics.csv` and `selections.csv` have 270 rows and the expected unique matrix.
- `analysis.md` numbers and conclusions match CSV/summary aggregates.

Expected validations:
- Lightweight strict JSON parse and CSV row count.
- Recompute headline aggregates from metrics/selections.
- Syntax check or AST parse for benchmark file if useful.

Acceptance criteria:
- If satisfied, explicitly say there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide concrete findings with severity and evidence.

Parallel/sequential notes:
- This review gates the next combined class/group strategy experiment.
