# Opt-In Label Studio Live Tests

The default test suite does not start Docker or contact Label Studio.

To run the managed Docker smoke test locally, provide explicit credentials and opt in:

```powershell
$env:ACTIVE_LEARNING_RUN_LIVE_LABEL_STUDIO = "1"
$env:ACTIVE_LEARNING_LABEL_STUDIO_USERNAME = "user@example.test"
$env:ACTIVE_LEARNING_LABEL_STUDIO_PASSWORD = "<password>"
$env:ACTIVE_LEARNING_LABEL_STUDIO_TOKEN = "<token>"
$env:ACTIVE_LEARNING_LABEL_STUDIO_PORT = "9091"
uv run pytest tests/test_live_label_studio_integration.py -q
```

The test is skipped unless `ACTIVE_LEARNING_RUN_LIVE_LABEL_STUDIO=1` is set. It also skips when Docker Compose or required credentials are unavailable.
