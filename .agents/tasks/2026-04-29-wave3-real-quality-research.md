# Task wave3-real-quality-research

## Context
Wave3 quality benchmarks still fail on capped Banking77 and DAIR.AI Emotion. Banking77 improved from full collapse but still has partial aliasing: `entropy` aliases `coreset_kcenter`, and `badge` aliases `class_group_balanced_entropy` in most groups. Cold-start fallback remapping appears in diagnostics.

## Goal
Research root causes and propose SDK/benchmark changes that improve real active-learning quality honestly without weakening gates. Do not edit files.

## In Scope
- Inspect strategy cold-start/fallback code, benchmark selection artifacts, quality gate logic, and model behavior.
- Determine whether fallback remapping is too aggressive, too long-lived, or missing probability-based scoring once a model is usable.
- Compare intended heuristics for many-class low-budget scenarios.
- Propose concrete changes and focused validation commands.

## Out Of Scope
- Do not modify files.
- Do not claim quality improvements without benchmark evidence.

## Acceptance Criteria
- Root cause hypotheses ranked by likelihood.
- Proposed implementation experiments.
- Exact tests/benchmarks to run after implementation.
