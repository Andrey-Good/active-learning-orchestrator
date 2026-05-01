Task ID: R48
Short name: audit mix scheduler
Relation to overall task: Read-only investigation of whether current `mode="mix"` scheduling behavior is a quality bottleneck for active-learning outcomes.

Assumptions and resolved ambiguities:
- Existing mix mode allocates counts by weights and executes strategy arms in sorted-name blocks.
- Previous benchmarks show mix strategies help early/rare-class behavior but can worsen group concentration.
- This is read-only research; do not edit files.

Goal and expected result:
- Audit current mix scheduler implementation and tests.
- Identify concrete algorithmic risks and whether a future SDK change should interleave arms, use weighted round-robin, or add group-aware de-duplication across arms.
- Provide a small, testable recommendation, not a broad rewrite.

Responsibility boundaries:
- Review scope:
  - `src/active_learning_sdk/engine.py`
  - `src/active_learning_sdk/configs.py`
  - tests that cover scheduler/mix behavior
  - benchmark analysis files only as evidence, not for editing
- Out of scope:
  - Any file edits
  - Running new benchmark jobs
  - SDK implementation changes

Review checks:
- How does mix allocate budgets and order arms?
- Does sorted-name block execution bias selection order or group concentration?
- Are strategy arms rerun on the remaining pool after each arm?
- Are allocation rounding and fallback deterministic and fair?
- Are there tests that lock in current behavior?
- What minimal future change would be non-breaking or explicitly breaking?

Expected validations:
- Read relevant files.
- Optionally run focused existing tests read-only if helpful.

Acceptance criteria:
- Final response lists concrete findings and one recommended next implementation experiment if warranted.
- If no scheduler issue is found, say so explicitly with evidence.

Parallel/sequential notes:
- Can run in parallel with W38 benchmark probe.
