Task ID: R62
Short name: Stage 0 readiness audit
Relation to overall task: Read-only audit for Stage 0 stabilization while dependency cleanup runs separately.

Goal and expected result:
- Audit whether current docs/package/benchmark state is ready for a Stage 0 baseline freeze after notebook core dependencies are removed.
- Identify any stale notebook/legacy benchmark references or missing validation commands.

Responsibility boundaries:
- Read-only scope:
  - `README.md`
  - `benchmarks/README.md`
  - `benchmarks/results/current_benchmark_report.md`
  - `docs/SDK_REAL_PRODUCT_ROADMAP.md`
  - `pyproject.toml` as context only
  - repository file existence checks
- Out of scope:
  - editing files
  - running long benchmarks
  - touching `src/**` or `tests/**`

Review checks:
- No active docs claim notebook benchmark entrypoints exist.
- No `.ipynb` files remain.
- No `experiment_runs.csv` remains.
- Stage 0 exit criteria are concrete and measurable.
- The current benchmark/reference artifacts needed for Stage 0 exist.

Expected validations:
- Search active docs for stale notebook/legacy artifact references.
- Check existence of benchmark result directories used in docs.
- Check current tests list and benchmark scripts exist.

Acceptance criteria:
- Final response lists blocking findings or explicitly states Stage 0 docs/artifacts are ready after dependency cleanup review.

Dependencies:
- Can run in parallel with W48 because it is read-only.
