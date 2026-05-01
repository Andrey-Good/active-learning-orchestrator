Task ID: W34
Short name: benchmark class-balanced entropy
Relation to overall task: Evaluate newly implemented `class_balanced_entropy` against existing baselines before deciding whether and how to further improve SDK acquisition methods.

Assumptions and resolved ambiguities:
- R42 accepted the SDK implementation of `class_balanced_entropy`.
- R43 accepted `mix_group_diverse_random` benchmark integration and artifacts.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Add `class_balanced_entropy` to benchmark strategy specs if it is not already present.
- Run a targeted comparative benchmark across all diagnostic datasets.
- Produce result artifacts and an honest `analysis.md` describing whether class balancing improves macro-F1 AULC, early macro-F1, rare recall, and group concentration.

Responsibility boundaries:
- In scope:
  - `benchmarks/sdk_first_benchmark.py`
  - `benchmarks/README.md` if documentation needs updating
  - `benchmarks/results/class_balanced_entropy/`
- Out of scope:
  - `src/**`
  - `tests/**`
  - root README
  - Docker/backends

Architectural constraints and forbidden actions:
- Do not change SDK implementation.
- Do not weaken validation/leakage checks.
- Do not overwrite existing benchmark result directories.
- Keep comparison matrix deterministic and comparable to earlier runs.

High-level execution plan:
- Register benchmark spec `class_balanced_entropy` with `SchedulerConfig(mode="single", strategy="class_balanced_entropy")`.
- Run:
  `uv run python benchmarks/sdk_first_benchmark.py --datasets grouped_duplicates,rare_class_trap,separable_topics --strategies random,entropy,group_diverse_entropy,class_balanced_entropy,mix_uncertainty_random,mix_group_diverse_random --seeds 13,17,23 --budgets 16,32,48,64,96 --output-dir benchmarks/results/class_balanced_entropy`
- Validate strict JSON parsing and expected row counts: 3 datasets x 6 strategies x 3 seeds x 5 budgets = 270 rows for metrics and selections.
- Write or update `analysis.md` with headline metrics and a clear conclusion.

Step -> verify:
- Add strategy spec -> verify CLI accepts the strategy.
- Run benchmark -> verify all artifacts exist.
- Validate artifacts -> verify strict JSON parse and row counts.
- Analyze -> verify reported numbers match CSV/summary aggregates.

Acceptance criteria:
- `benchmarks/results/class_balanced_entropy/` contains manifest, metrics, selections, summary, validation, and analysis.
- Analysis states whether the new strategy is better, worse, or mixed on each dataset.
- The conclusion distinguishes primary metric (macro-F1 AULC) from guardrails (rare recall, group HHI/top-group fraction, early metric).

Expected tests and validations:
- `python -m py_compile benchmarks/sdk_first_benchmark.py`
- Benchmark command above
- Strict JSON parse for result JSON files
- CSV row count validation

Dependencies:
- Depends on R42 and R43 acceptance.

Parallel/sequential notes:
- This task is sequentially after R42/R43.
