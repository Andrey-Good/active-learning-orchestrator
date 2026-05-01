Task ID: W41
Short name: benchmark mix interleaved scheduler
Relation to overall task: Evaluate accepted `mix_interleaved` scheduler against current block-based mix after R51 approved implementation.

Assumptions and resolved ambiguities:
- R51 accepted `mix_interleaved` implementation and observability.
- Current block-based mixes improve some quality metrics but regress group concentration.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Add benchmark strategy specs for interleaved equivalents of existing hybrid mixes.
- Run a targeted comparative benchmark across all diagnostic datasets.
- Produce result artifacts and honest analysis of whether interleaving improves group concentration while preserving macro-F1 AULC, early macro-F1, and rare recall.

Responsibility boundaries:
- In scope:
  - `benchmarks/sdk_first_benchmark.py`
  - `benchmarks/README.md` if documentation needs updating
  - `benchmarks/results/mix_interleaved_probe/`
- Out of scope:
  - `src/**`
  - `tests/**`
  - root README
  - Docker/backends

Architectural constraints and forbidden actions:
- Do not change SDK source.
- Do not weaken validation/leakage checks.
- Do not overwrite existing result directories.
- Keep comparison deterministic and comparable to W38.

Suggested strategy specs:
- `mix_interleaved_class_group_random`: `SchedulerConfig(mode="mix_interleaved", mix={"class_group_balanced_entropy": 0.7, "random": 0.3})`
- `mix_interleaved_class_group_margin_random`: `SchedulerConfig(mode="mix_interleaved", mix={"class_group_balanced_entropy": 0.4, "margin": 0.3, "random": 0.3})`

High-level execution plan:
- Add suggested strategy specs.
- Run:
  `uv run python benchmarks/sdk_first_benchmark.py --datasets grouped_duplicates,rare_class_trap,separable_topics --strategies class_group_balanced_entropy,mix_class_group_random,mix_class_group_margin_random,mix_interleaved_class_group_random,mix_interleaved_class_group_margin_random --seeds 13,17,23 --budgets 16,32,48,64,96 --output-dir benchmarks/results/mix_interleaved_probe`
- Validate strict JSON and expected row counts: 3 datasets x 5 strategies x 3 seeds x 5 budgets = 225 rows for metrics and selections.
- Inspect scheduler snapshots for `mix_interleaved` observability fields.
- Write/ensure `analysis.md`.

Step -> verify:
- Specs added -> verify CLI accepts them.
- Benchmark run -> verify artifacts.
- Validation -> verify strict JSON, row counts, leakage checks.
- Analysis -> verify aggregates match CSV/summary.

Acceptance criteria:
- Artifacts exist under `benchmarks/results/mix_interleaved_probe/`.
- Analysis compares interleaved vs block-mix equivalents on macro-F1 AULC, early macro-F1, rare recall, group HHI/top-group fraction.
- Analysis states whether the scheduler change is accepted, rejected, or mixed.

Expected tests and validations:
- `python -m py_compile benchmarks/sdk_first_benchmark.py`
- Benchmark command above
- Strict JSON parse
- CSV row count validation
- Scheduler snapshot spot check for new fields

Dependencies:
- Sequentially after R51.

Parallel/sequential notes:
- Benchmark-only task. SDK source must not be touched.
