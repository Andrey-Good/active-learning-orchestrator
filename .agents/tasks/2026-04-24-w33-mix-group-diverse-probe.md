Task ID: W33
Short name: mix group-diverse probe
Relation to overall task: Test reviewer-suggested hypothesis that replacing entropy with group-diverse entropy inside the existing mix scheduler may preserve uncertainty quality while reducing grouped duplicate concentration.

Assumptions and resolved ambiguities:
- W31 showed `group_diverse_entropy` improves concentration metrics but is not always best in macro-F1.
- Existing scheduler mix mode should be able to combine `group_diverse_entropy`, `margin`, and `random` without SDK source changes.
- This worker is not alone in the codebase. Another worker may edit SDK strategy/source files in parallel. Do not touch those files.

Goal and expected result:
- Add a benchmark-only strategy spec for a group-diverse mix, preferably `mix_group_diverse_random`.
- Run a targeted benchmark against `grouped_duplicates` and `rare_class_trap` with the same seeds/budgets used in W31.
- Produce result artifacts and a concise analysis comparing against `random`, `entropy`, `group_diverse_entropy`, and `mix_uncertainty_random`.

Responsibility boundaries:
- In scope:
  - `benchmarks/sdk_first_benchmark.py`
  - `benchmarks/README.md` only if needed to document the new benchmark strategy
  - `benchmarks/results/mix_group_diverse_probe/`
- Out of scope:
  - `src/active_learning_sdk/**`
  - `tests/**`
  - README root
  - Docker/backends/docs outside benchmarks

Architectural constraints and forbidden actions:
- Do not change SDK implementation.
- Do not weaken existing leakage validation or strict JSON behavior.
- Keep benchmark additions deterministic and comparable to previous W31 run.
- Do not overwrite existing benchmark result directories.

High-level execution plan:
- Add strategy spec using `SchedulerConfig(mode="mix", mix={"group_diverse_entropy": 0.4, "margin": 0.3, "random": 0.3})` or justify a different weight.
- Run targeted benchmark:
  `uv run python benchmarks/sdk_first_benchmark.py --datasets grouped_duplicates,rare_class_trap --strategies random,entropy,group_diverse_entropy,mix_uncertainty_random,mix_group_diverse_random --seeds 13,17,23 --budgets 16,32,48,64,96 --output-dir benchmarks/results/mix_group_diverse_probe`
- Validate JSON artifacts parse and row counts are expected.
- Write/ensure `analysis.md` accurately states whether the hypothesis improved metrics.

Step -> verify:
- Strategy spec added -> verify CLI accepts it.
- Benchmark run -> verify metrics/selections/summary/validation artifacts exist.
- Analysis -> verify aggregates in `analysis.md` match CSV/summary data.

Acceptance criteria:
- Targeted benchmark artifacts exist under a new output directory.
- The new mix is compared numerically against current baselines.
- The conclusion is honest if it fails or only improves secondary metrics.

Expected tests and validations:
- Benchmark command above.
- Strict JSON parse of result JSON files.
- Basic CSV row count check.

Dependencies:
- Depends on W31 `group_diverse_entropy` being registered in SDK and benchmark harness.
- Independent of W32.

Parallel/sequential notes:
- Can run in parallel with W32 because write scopes do not overlap.
