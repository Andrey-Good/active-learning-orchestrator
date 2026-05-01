Task ID: W37
Short name: benchmark class-group balanced entropy
Relation to overall task: Evaluate the hybrid `class_group_balanced_entropy` strategy against accepted baselines after R46 approved the SDK implementation.

Assumptions and resolved ambiguities:
- R46 accepted `class_group_balanced_entropy`.
- Current accepted strategy baselines include `random`, `entropy`, `group_diverse_entropy`, `class_balanced_entropy`, `mix_uncertainty_random`, and `mix_group_diverse_random`.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Add `class_group_balanced_entropy` to benchmark strategy specs if absent.
- Run a targeted comparative benchmark across all diagnostic datasets.
- Produce result artifacts and an honest `analysis.md` describing whether the hybrid improves macro-F1 AULC, early macro-F1, rare recall, and group concentration.

Responsibility boundaries:
- In scope:
  - `benchmarks/sdk_first_benchmark.py`
  - `benchmarks/README.md` if documentation needs updating
  - `benchmarks/results/class_group_balanced_entropy/`
- Out of scope:
  - `src/**`
  - `tests/**`
  - root README
  - Docker/backends

Architectural constraints and forbidden actions:
- Do not change SDK implementation.
- Do not weaken validation/leakage checks.
- Do not overwrite existing result directories.
- Keep comparison matrix deterministic and comparable to W34.

High-level execution plan:
- Register benchmark spec `class_group_balanced_entropy` with `SchedulerConfig(mode="single", strategy="class_group_balanced_entropy")`.
- Run:
  `uv run python benchmarks/sdk_first_benchmark.py --datasets grouped_duplicates,rare_class_trap,separable_topics --strategies random,entropy,group_diverse_entropy,class_balanced_entropy,class_group_balanced_entropy,mix_uncertainty_random,mix_group_diverse_random --seeds 13,17,23 --budgets 16,32,48,64,96 --output-dir benchmarks/results/class_group_balanced_entropy`
- Validate strict JSON parsing and expected row counts: 3 datasets x 7 strategies x 3 seeds x 5 budgets = 315 rows for metrics and selections.
- Write or ensure `analysis.md` with headline metrics and a clear conclusion.

Step -> verify:
- Add strategy spec -> verify CLI accepts it.
- Run benchmark -> verify all artifacts exist.
- Validate artifacts -> verify strict JSON parse and row counts.
- Analyze -> verify reported numbers match CSV/summary aggregates.

Acceptance criteria:
- `benchmarks/results/class_group_balanced_entropy/` contains manifest, metrics, selections, summary, validation, and analysis.
- Analysis states whether the hybrid is better, worse, or mixed on each dataset.
- The conclusion separates primary metric from guardrails.

Expected tests and validations:
- `python -m py_compile benchmarks/sdk_first_benchmark.py`
- Benchmark command above
- Strict JSON parse for result JSON files
- CSV row count validation

Dependencies:
- Sequentially after R46.

Parallel/sequential notes:
- Benchmark-only task; SDK source must not be touched.
