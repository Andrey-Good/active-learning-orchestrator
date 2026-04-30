# W06 - Baseline Sweep Runner

## Relation to Overall Task
This worker implements the next measurable research loop: quickly test whether the current baseline failure is due to model choice, synthetic task difficulty, budget threshold, or seed instability.

## Assumptions and Resolved Ambiguities
- Engine round/budget semantics have been fixed and independently reviewed.
- The current `synthetic + bert_tiny` artifact is honest but fails random-baseline acceptance at 48 labels.
- We need new benchmark artifacts before optimizing strategy behavior further.
- This task must stay focused on existing methods and benchmark quality; do not add BADGE or new acquisition algorithms.

## Goal
Extend the learning-curve benchmark tooling to run a compact baseline stability sweep and write uniform CSV/JSON/Markdown artifacts that compare candidate baselines.

## Responsibility Boundaries
- Own benchmark automation and generated baseline-sweep artifacts only.
- Keep SDK source changes out of scope unless an unavoidable benchmark blocker is discovered; if so, stop and report.

## In Scope
- Modify `benchmarks/run_learning_curve_experiments.py` as needed.
- Modify `lab/learning_curve_lab.ipynb` only if needed to expose or document the new runner usage.
- Generate artifacts under `benchmarks/results/learning_curves/`.
- Add support for a compact sweep over candidate dataset/model/config variants.
- Compute metrics that help choose a baseline:
  - random accuracy/macro-F1 at 48 and 80 labels;
  - AULC macro-F1/accuracy;
  - seed standard deviation for random at key budgets;
  - slope from first to last budget;
  - best heuristic lift vs random;
  - runtime summary.

## Out of Scope
- Do not change `src/active_learning_sdk/**`.
- Do not change Docker/Label Studio.
- Do not add new strategies.
- Do not turn this into a large full benchmark matrix.

## Files/Modules May Change
- `benchmarks/run_learning_curve_experiments.py`
- `lab/learning_curve_lab.ipynb`
- New/updated files under `benchmarks/results/learning_curves/`

## Files/Areas Must Not Touch
- `src/active_learning_sdk/**`
- `README.md`
- Docker files
- unrelated benchmark scripts unless absolutely necessary

## Architectural Constraints and Forbidden Actions
- Preserve deterministic seed-order check behavior.
- Do not overwrite the existing baseline artifact unless intentional and documented.
- Keep generated artifacts small enough for repository review.
- If a candidate fails, record the failure honestly.
- Prefer reusable functions rather than copy-pasted one-off experiment code.

## High-Level Execution Plan
- Add CLI mode or options for a compact baseline sweep -> verify default existing behavior still works.
- Add candidate configurations -> verify they can alter model key and synthetic difficulty without notebook edits where possible.
- Run the sweep -> verify artifacts have no duplicate keys and all rows are `ok` or have explicit failure reasons.
- Summarize candidate quality -> verify the summary identifies whether any baseline passes acceptance.
- Update the notebook with an executable regeneration cell if needed -> verify notebook JSON is valid.

## Step -> Verify Plan
- Implement sweep configuration support -> run `python -m py_compile benchmarks/run_learning_curve_experiments.py`.
- Run default learning-curve command -> verify current artifacts still generate.
- Run baseline sweep command -> verify output CSV/JSON/Markdown are created.
- Run seed-order reproducibility check -> verify deterministic rows are identical ignoring duration.
- Validate artifacts -> verify row counts, uniqueness, required columns, and budget/strategy/seed coverage.

## Acceptance Criteria
- Existing `python benchmarks/run_learning_curve_experiments.py` behavior still works.
- A new compact sweep can be run via a documented command.
- Sweep artifacts include enough metrics to choose or reject a baseline.
- The final worker report includes the top candidate and why it passed/failed.
- All validations run and are reported.

## Expected Tests and Validations
- `python -m py_compile benchmarks/run_learning_curve_experiments.py`
- `.venv\\Scripts\\python.exe benchmarks\\run_learning_curve_experiments.py`
- New sweep command
- `.venv\\Scripts\\python.exe benchmarks\\run_learning_curve_experiments.py --check-seed-order`
- Artifact integrity checks
- Notebook JSON validation if modified

## Dependencies
- Depends on W01/W03/W05 engine fixes being present.

## Parallel/Sequential Notes
- Can run in parallel with R10.
- Must be independently reviewed after completion.
