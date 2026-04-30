# Reset Benchmark Layer Plan

## Objective
Remove the old notebook/test/benchmark layer and replace it with a clean SDK-first benchmark harness that can validly drive active-learning algorithm improvements.

## Scope
Remove stale artifacts:
- notebooks (`*.ipynb`);
- old `benchmarks/**`;
- old `tests/**`;
- old generated experiment CSVs in lab/root if they are benchmark outputs.

Preserve product code:
- `src/active_learning_sdk/**`;
- docs unless explicitly replaced later;
- Docker/backend product files.

## New Benchmark Requirements
The new benchmarks must:
- call SDK strategy/scheduler code directly, not notebook-local selection functions;
- separate dataset seed, initial labeled seed, acquisition seed, and model seed;
- evaluate label-budgeted learning curves;
- include multiple dataset regimes where different AL behavior is expected;
- compare against random with enough seeds;
- report quality, label efficiency, stability, selected-label skew, redundancy/diversity, and runtime;
- be runnable without notebooks;
- be deterministic and artifact-based.

## Algorithm Improvement Loop
After benchmark harness exists:
1. run baseline over existing strategies;
2. identify failure mode from metrics/diagnostics;
3. implement one SDK improvement at a time;
4. rerun relevant benchmark slice;
5. promote only if metrics pass predefined gates;
6. independently review each implementation and benchmark conclusion.

## Current Hypotheses To Test Later
- Existing uncertainty formulas are OK but benchmark protocol was invalid.
- Batch uncertainty needs diversity/coverage to beat random reliably.
- Initial seed policy strongly affects neural/small-data AL quality.
- Probability calibration may affect least-confidence and entropy.
- SDK random/mix reproducibility improvements are necessary but not enough.

## Post-Harness Improvement Backlog
These are not implementation approvals yet; each item needs its own worker/reviewer cycle and benchmark evidence.

- Initial seed/import-labels API: add a public way to mark an initial labeled batch before active acquisition. Metric: AULC and variance at budgets 16-48 versus first-round uniform uncertainty or deterministic tie selection.
- Diversity batch selection: implement an actually usable embedding diversity strategy instead of the current rejected `coreset_kcenter` placeholder. Metric: duplicate/group concentration, macro-F1 lift on redundant-cluster datasets, and runtime overhead.
- Hybrid uncertainty + diversity: add a strategy that scores uncertainty but penalizes near-duplicates inside the same acquisition batch. Metric: lift versus pure uncertainty on `synth_redundant_clusters` and no regression on `synth_boundary_help`.
- Class-balance-aware acquisition: add a conservative predicted-class balancing option for rare-class and imbalance regimes. Metric: rare recall, macro-F1, and guardrail against accuracy collapse.
- Bandit scheduler: replace first-arm placeholder with deterministic UCB-style arm selection and reward traces. Metric: best-arm convergence, no worse than random/mix on average, and explicit regret summary.
- Better stop criteria: record stop decisions with metric traces and compare saved labels/runtime against quality loss. Metric: saved budget, AULC delta, final macro-F1 delta.
- Calibration-aware uncertainty: optionally calibrate or temper probabilities before entropy/least-confidence. Metric: ECE/Brier/NLL changes and downstream acquisition lift.

## Promotion Gates
- A method can be promoted only if it beats or matches random on the dataset regime it is designed for across multiple seeds.
- A method can still be useful if it fails on a regime, but the benchmark summary must name that failure mode rather than hide it.
- Runtime overhead must be reported separately from quality gains.
- Any SDK API addition must have tests before final productization, even if benchmark code is not packaged as tests.
