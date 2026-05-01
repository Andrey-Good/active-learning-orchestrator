# 2026-04-29 Wave6 W06C Quality Matrix Stress

## Task Identifier

W06C-QUALITY-MATRIX-STRESS.

## Context

Prior waves found real and synthetic quality weaknesses, then subsequent fixes reportedly improved several gates. This worker runs a fresh documented quality matrix to find low metrics, regressions versus random, selection collapse, and reproducibility issues.

## Goal

Stress the SDK's documented active-learning quality behavior across datasets, strategies, budgets, seeds, and model settings.

## Ownership

May write only under `.agents/tmp/blackbox_stress_wave6/w06c_quality/`.

## In Scope

- Execute documented benchmark entrypoints from README/benchmark docs without reading benchmark implementation source.
- Use synthetic diagnostic datasets and capped real datasets where feasible.
- Include multiple strategies: random, uncertainty, class/group balance, embedding/diversity, BADGE/adaptive/mix where available.
- Analyze macro-F1, AULC, non-loss rate, zero-recall fraction, runtime, selected-count, and selection-collapse indicators from generated artifacts.
- Write `findings.md`, copied/summarized command logs, and metric summaries.

## Out Of Scope

- Reading benchmark implementation source or SDK source.
- Changing benchmark source.
- Making broad scientific claims beyond the tested bounded matrices.
- Long unbounded full-corpus experiments.

## Important Constraints

- Label all real-dataset runs as capped diagnostics unless they meet the documented standard evidence contract.
- A strategy losing to random is a quality/product finding, not automatically an SDK correctness bug.
- Reject false positives caused by requested budgets below seed size, documented HF scaffold limits, or intentionally basic bandit behavior.

## Execution Plan

1. Read README benchmark command and evidence rules only.
2. Run one fast synthetic matrix and one or more bounded real matrices.
3. Generate quality reports when documented.
4. Summarize worst regressions, passing strategies, and reproducibility gaps.

## Acceptance Criteria

- At least one synthetic and one real/capped dataset family attempted.
- Metrics are compared against matched random baseline.
- Any quality gate failure includes the exact command and artifact directory.
