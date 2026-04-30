# Task r107: Review Cold-Start Quality Guardrail

## Context
The orchestrator implemented SDK changes to improve active-learning metrics:
- uncertainty strategies now switch to deterministic exploration when probability support covers too little of the label schema;
- `density_weighted_diversity` now estimates density without allocating an enormous 3D pairwise tensor;
- focused tests were added.

## Goal
Review the implementation for correctness, product quality, architectural fit, and test adequacy.

## Responsibility Boundaries
This is a read-only review.

## In Scope
- `src/active_learning_sdk/strategies/uncertainty.py`
- `src/active_learning_sdk/strategies/embedding.py`
- `tests/test_class_balanced_entropy_strategy.py`
- `tests/test_embedding_strategies.py`

## Out of Scope
- Do not edit files.
- Do not review unrelated dirty worktree changes.

## Review Questions
- Is the cold-start guardrail deterministic and non-leaky?
- Does it preserve entropy's minimal capability contract when embeddings are unavailable?
- Is switching all uncertainty variants to novelty exploration under sparse support defensible?
- Is the density memory fix correct and safe for large pools?
- Are tests sufficient for edge cases?

## Acceptance Criteria
- Final answer lists findings with severity and file/line references.
- If no blocking findings, state residual risks.

## Dependencies
Depends on the orchestrator's local patch.
