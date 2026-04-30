# Vector Temperature Scaling Plan

## Goal
- Add scalar/vector temperature scaling mode selection to `lab/active_learning_lab.ipynb`.
- Keep the existing notebook structure intact and avoid unrelated edits.

## Scope
- Add one config parameter for calibration mode.
- Extend `TemperatureScaling` to branch between scalar and vector temperature fitting.
- Pass the selected mode when creating the calibrator in local and SDK runners.
- Persist the selected mode in the result row.
- Run a lightweight smoke check after changes.

## Steps
1. Inspect current config, class, and runner sections in the lab notebook.
2. Add a calibration mode config value with validation and optional env override.
3. Update `TemperatureScaling`:
   - store scaling mode;
   - fit scalar or vector temperature depending on mode;
   - keep transform and metrics compatible with both modes.
4. Update local and SDK experiment runners to instantiate the calibrator with the configured mode.
5. Add the selected calibration mode to the saved result row.
6. Validate notebook JSON and run a lightweight synthetic smoke test.

## Guardrails
- Do not change training logic beyond post-hoc calibration mode support.
- Do not add extra calibration methods beyond scalar/vector temperature scaling.
- Do not restructure unrelated notebook cells.
- Keep defaults conservative: scalar remains the default mode.
