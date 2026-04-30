# R26 - Design New SDK-First Benchmarks

## Relation to Overall Task
Read-only design task before implementing the new benchmark harness.

## Goal
Design benchmark scenarios and metrics that correctly evaluate active-learning quality and guide SDK improvements.

## Requirements
The design must cover:
- random, entropy, margin, least-confidence, mix scheduler;
- initial seed effects;
- batch diversity/redundancy;
- label-budgeted performance;
- stability over seeds;
- calibration/probability quality;
- runtime and acquisition overhead;
- datasets where uncertainty should help and where it should fail.

## In Scope
- Propose benchmark datasets, preferably fast deterministic synthetic text datasets plus optional sklearn/HF extensions.
- Define exact metrics and pass/fail gates.
- Define artifact schema.
- Define small default run and larger research run.
- Explain how benchmarks call SDK code directly.

## Out of Scope
- Do not edit files.
- Do not run experiments.

## Acceptance Criteria
- Final report gives a concrete implementable benchmark spec.
- Design explicitly fixes old benchmark flaws.
