Task ID: R49
Short name: review hybrid mix probe
Relation to overall task: Independent review of W38 benchmark-only hybrid mix probe before deciding whether to change mix scheduler behavior.

Assumptions and resolved ambiguities:
- W38 claims hybrid mixes improve some metrics but regress group concentration.
- R48 independently found scheduler mix has block/order/group-awareness issues.
- Reviewer is read-only and must not edit files.

Goal and expected result:
- Verify W38 benchmark strategy specs, artifacts, row counts, strict JSON validity, scheduler snapshots, and analysis headline numbers.
- Explicitly state whether there are no remaining in-scope defects, blocking risks, blocking questions, or required improvements.

Responsibility boundaries:
- Review scope:
  - `benchmarks/sdk_first_benchmark.py`
  - `benchmarks/README.md`
  - `benchmarks/results/hybrid_mix_probe/**`
- Out of scope:
  - `src/**`
  - `tests/**`
  - root README
  - Docker/backends

Review checks:
- `mix_class_group_random` and `mix_class_group_margin_random` specs use intended configs.
- Benchmark validation/leakage checks were not weakened.
- JSON artifacts parse strictly.
- `metrics.csv` and `selections.csv` have 360 rows and expected unique matrix.
- Scheduler snapshots for new mix rows confirm intended mix configs.
- `analysis.md` headline metrics and conclusion match recomputed aggregates.

Expected validations:
- Strict JSON parse.
- CSV row-count/unique-key validation.
- Recompute macro-F1 AULC, budget-16 macro-F1, rare recall AULC, group HHI/top-group fraction.
- Inspect scheduler snapshots for new mix strategies.

Acceptance criteria:
- If satisfied, explicitly say there are no remaining in-scope defects, blocking risks, blocking questions, or required improvement requests.
- If not satisfied, provide concrete findings with severity and evidence.

Parallel/sequential notes:
- This review gates any scheduler implementation experiment.
