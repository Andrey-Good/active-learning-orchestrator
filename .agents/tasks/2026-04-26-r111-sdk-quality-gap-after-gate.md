# Task r111: SDK Quality Gap After Gate

## Context
After multi-seed/multi-budget quality-gate results, the SDK may still fail criteria. This task is for read-only diagnosis of remaining strategy weaknesses.

## Goal
Given benchmark outputs, identify why any strategy/dataset fails and propose targeted SDK improvements.

## Responsibility Boundaries
Read-only unless explicitly reassigned.

## In Scope
- Benchmark result artifacts.
- Strategy code and selection diagnostics.

## Out of Scope
- No edits.
- No oracle-label leakage recommendations.

## Acceptance Criteria
- Findings map failed metrics to likely causes and concrete fixes.

## Dependencies
Depends on quality-gate outputs.
