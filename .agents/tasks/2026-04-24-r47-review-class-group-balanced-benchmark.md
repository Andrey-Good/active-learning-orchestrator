Task ID: R47
Short name: review class-group balanced benchmark
Relation to overall task: Independent review of W37 benchmark results because the hybrid strategy appears to improve all primary diagnostic metrics.

Assumptions and resolved ambiguities:
- W37 claims `class_group_balanced_entropy` ranks best on all diagnostic datasets by mean macro-F1 AULC and ties best group concentration guardrails.
- Reviewer is read-only and must not edit files.

Goal and expected result:
- Verify benchmark spec, artifacts, row counts, strict JSON validity, scheduler snapshots, and `analysis.md` headline numbers.
- Pay special attention to whether the strong result could be caused by benchmark leakage, weakened validation, wrong strategy mapping, stale artifacts, or aggregation mistakes.
- Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvements.

Responsibility boundaries:
- Review scope:
  - `benchmarks/sdk_first_benchmark.py`
  - `benchmarks/README.md`
  - `benchmarks/results/class_group_balanced_entropy/**`
- Out of scope:
  - `src/**`
  - `tests/**`
  - root README
  - Docker/backends

Review checks:
- Strategy spec uses `SchedulerConfig(mode="single", strategy="class_group_balanced_entropy")`.
- No benchmark validation/leakage checks were weakened.
- JSON artifacts parse strictly.
- `metrics.csv` and `selections.csv` have 315 rows and expected unique matrix.
- Scheduler snapshots for hybrid rows confirm the intended strategy.
- `validation.json` leakage checks remain clean.
- `analysis.md` headline metrics match recomputed aggregates.
- Conclusions are honest about early-budget and synthetic-only caveats.

Expected validations:
- Strict JSON parse.
- CSV row-count and unique-key validation.
- Recompute macro-F1 AULC, budget-16 macro-F1, rare recall, group HHI/top-group fraction.
- Inspect scheduler snapshots for hybrid rows.

Acceptance criteria:
- If satisfied, explicitly say there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide concrete findings with severity and evidence.

Parallel/sequential notes:
- This review gates accepting the hybrid as current best SDK strategy.
