Task ID: W35
Short name: fix benchmark SDK gap metadata
Relation to overall task: Resolve R44 required metadata correction so benchmark artifacts accurately describe current SDK capabilities before further benchmark-driven decisions.

Assumptions and resolved ambiguities:
- R44 found that benchmark metadata still says `ActiveLearningProject` lacks a public initial-label import API.
- The SDK now has `ActiveLearningProject.import_labels(...)`, and project smoke uses it.
- This worker is not alone in the codebase. Do not revert or overwrite unrelated edits.

Goal and expected result:
- Update benchmark metadata so it no longer reports the stale `import_labels` gap.
- Regenerate or patch affected benchmark artifacts for `benchmarks/results/class_balanced_entropy/` so manifest/summary metadata is accurate.
- Keep benchmark numeric results unchanged unless a rerun is necessary.

Responsibility boundaries:
- In scope:
  - `benchmarks/sdk_first_benchmark.py`
  - `benchmarks/results/class_balanced_entropy/manifest.json`
  - `benchmarks/results/class_balanced_entropy/summary.json`
  - Any directly derived benchmark metadata files if the harness writes the same stale note elsewhere
- Out of scope:
  - `src/**`
  - `tests/**`
  - root README
  - benchmark numeric CSV data unless regeneration is unavoidable

Architectural constraints and forbidden actions:
- Do not change SDK source.
- Do not weaken benchmark validations.
- Do not falsify history; describe current SDK limitations accurately.
- Prefer regenerating artifacts with the corrected harness if fast; otherwise patch only metadata fields and validate strict JSON.

High-level execution plan:
- Locate the stale metadata in `benchmarks/sdk_first_benchmark.py`.
- Replace it with accurate current metadata, likely noting that public seed import is available via `ActiveLearningProject.import_labels`.
- Update affected class-balanced result JSON metadata.
- Validate strict JSON parsing and that numeric row counts remain unchanged.

Step -> verify:
- Metadata source fixed -> verify no stale phrase remains in benchmark harness/results.
- Artifacts updated -> verify JSON parses.
- CSVs untouched or row counts unchanged -> verify metrics/selections still have 270 rows.

Acceptance criteria:
- No benchmark metadata claims that `ActiveLearningProject` lacks public initial-label import.
- `manifest.json` and `summary.json` for class-balanced results are consistent with current SDK behavior.
- Validation command output is reported.

Expected tests and validations:
- Search for the stale phrase.
- Strict JSON parse affected JSON files.
- CSV row count check for `benchmarks/results/class_balanced_entropy/metrics.csv` and `selections.csv`.

Dependencies:
- Fixes R44 finding.

Parallel/sequential notes:
- This is required before accepting W34 and before using those artifacts in final docs.
