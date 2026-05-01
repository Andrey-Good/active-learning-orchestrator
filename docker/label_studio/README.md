Managed Label Studio assets for the SDK live here.

`docker-compose.yml` starts Label Studio behind a tiny nginx reverse proxy so managed mode can expose a single stable host port.

Packaged managed mode requires explicit credential secrets. The compose file fails fast unless these variables are provided by the SDK runtime or by you for manual Docker use:

- `LABEL_STUDIO_HOST_PORT`
- `LABEL_STUDIO_USERNAME`
- `LABEL_STUDIO_PASSWORD`
- `LABEL_STUDIO_USER_TOKEN`

For SDK-managed packaged mode, set `ACTIVE_LEARNING_LABEL_STUDIO_USERNAME` and `ACTIVE_LEARNING_LABEL_STUDIO_PASSWORD`, and provide the token through `LabelBackendConfig.api_token` or `ACTIVE_LEARNING_LABEL_STUDIO_TOKEN`. For manual local testing with this compose file, export the `LABEL_STUDIO_*` variables directly before running `docker compose up -d`.
