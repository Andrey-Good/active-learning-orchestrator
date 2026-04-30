# Task: Research Optimization Orchestration Plan

## Overall Objective

Turn the existing SDK and benchmark surface into a disciplined experimental system before adding major new active learning methods.

The immediate goal is not to add BADGE, BALD, or large diversity methods yet. The goal is to take everything that already affects training quality and make it measurable, comparable, and improvable through controlled experiments.

## Boundaries

In scope:

- existing SDK selection strategies;
- scheduler behavior;
- SDK vs local loop semantics;
- dataset/model benchmark setup;
- metrics and learning-curve reporting;
- baseline dataset/model design;
- notebook/script structure for repeatable experiments;
- tests needed to make current behavior trustworthy.

Out of scope for this first research cycle:

- implementing BADGE;
- implementing k-center;
- implementing new production model adapters;
- rewriting Label Studio integration;
- large public API redesign.

## Decomposition

Parallel research tasks:

1. Strategy and scheduler audit.
2. Engine and SDK loop semantics audit.
3. Benchmark and metric audit.
4. Baseline dataset/model design.

Sequential implementation after research:

1. Create a unified research notebook/runner template.
2. Add baseline dataset/model where random learns reliably.
3. Add learning-curve metrics and budget-efficiency metrics.
4. Run first controlled experiments.
5. Review and iterate on the first improvement.

## Scientific Method For Each Improvement

Each improvement must follow this cycle:

1. Define the target quality dimension.
2. Define one or more metrics.
3. Capture a baseline.
4. State hypotheses.
5. Change one factor at a time where possible.
6. Re-run experiments under the same budget.
7. Compare against baseline and random.
8. Keep the change only if it improves a clearly defined metric or reveals a useful negative result.

## Candidate Quality Dimensions

- final accuracy;
- final macro-F1;
- area under learning curve;
- labels needed to reach a target score;
- runtime per acquisition round;
- training runtime;
- SDK overhead over local loop;
- stability across seeds;
- calibration ECE;
- class balance of selected labels;
- duplicate/near-duplicate rate in selected batches;
- percentage of useful rounds before plateau.

## Initial Acceptance Criteria

- A written list exists of all current SDK surfaces worth optimizing.
- A benchmark/research design exists before large implementation.
- At least one baseline model/dataset pair is identified where random learns reliably.
- At least one new notebook or script template is created for controlled experiments.
- All claims remain backed by generated artifacts.
