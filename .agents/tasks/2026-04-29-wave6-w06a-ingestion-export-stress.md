# 2026-04-29 Wave6 W06A Ingestion And Export Stress

## Task Identifier

W06A-INGESTION-EXPORT-STRESS.

## Context

Prior black-box reports list DataFrame, CSV, and Parquet ingestion as residual risk. The public docs say dataset providers can come from `pandas.DataFrame`, CSV, Parquet, or custom providers. This worker validates that public ingestion and export behavior is robust without reading SDK source.

## Goal

Exercise SDK public ingestion/export paths across normal and adversarial tabular/text payloads.

## Ownership

May write only under `.agents/tmp/blackbox_stress_wave6/w06a_ingestion/`.

## In Scope

- Build fresh black-box scripts using public SDK imports.
- Test provider-style, DataFrame, CSV, and Parquet inputs where public docs make them discoverable.
- Stress duplicate IDs, missing text, non-string IDs, extra payload columns, meta/group drift, Unicode/control characters, empty strings, long strings, and label/export formats.
- Compare expected public exception taxonomy with observed exceptions.
- Produce `results.json`, `findings.md`, and any generated data files.

## Out Of Scope

- Reading `src/active_learning_sdk/**`.
- Reading repository `tests/**`.
- Modifying SDK, docs, benchmarks, or packaging files.
- Large unbounded dataset downloads.

## Important Constraints

- Treat the SDK as a black box.
- Prefer documented root/package public imports; if an import path is ambiguous, record the ambiguity as a usability observation rather than guessing internals.
- If a case violates a documented precondition, classify it as guardrail/taxonomy behavior, not as a correctness bug.

## Execution Plan

1. Read only README/contracts sections needed for dataset and export API signatures.
2. Create a standalone stress script under the owned artifact directory.
3. Run the matrix and capture command output.
4. Write concise findings with accepted/rejected candidate issues.

## Acceptance Criteria

- At least 20 ingestion/export scenarios run.
- Normal documented flows pass or are reported.
- Invalid flows report observed public exception category.
- Artifact paths and exact replay commands are included.
