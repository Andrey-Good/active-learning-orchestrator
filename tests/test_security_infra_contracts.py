from __future__ import annotations

import io
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urllib_error

import pytest

from active_learning_sdk.backends import label_studio as label_studio_module
from active_learning_sdk.backends import managed_docker as managed_docker_module
from active_learning_sdk.backends.label_studio import (
    _LabelStudioApiError,
    _LabelStudioHttpClient,
    LabelStudioBackend,
)
from active_learning_sdk.backends.managed_docker import ManagedLabelStudioRuntime
from active_learning_sdk.configs import AnnotationPolicy, LabelBackendConfig, LabelSchema
from active_learning_sdk.exceptions import InfrastructureError


class _FakeUrlResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeUrlResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class _StaticTaskClient:
    def __init__(self, task: Mapping[str, Any]) -> None:
        self._task = dict(task)

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> Any:
        del query, payload
        if method == "GET" and path == "/api/tasks/100/":
            return dict(self._task)
        raise AssertionError(f"unexpected request: {method} {path}")


def _managed_config(**overrides: Any) -> LabelBackendConfig:
    values: dict[str, Any] = {
        "backend": "label_studio",
        "mode": "managed_docker",
        "managed_port": 9091,
    }
    values.update(overrides)
    return LabelBackendConfig(**values)


def _external_config() -> LabelBackendConfig:
    return LabelBackendConfig(
        backend="label_studio",
        mode="external",
        url="http://label-studio.local",
        api_token="token",
        project_id=10,
    )


def test_http_client_does_not_retry_non_idempotent_post_after_transient_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_urlopen(request: Any, timeout: int) -> _FakeUrlResponse:
        del timeout
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib_error.HTTPError(
                request.full_url,
                500,
                "server error after possible side effect",
                {},
                io.BytesIO(b"created task but returned 500"),
            )
        return _FakeUrlResponse(b'{"id": 101}')

    monkeypatch.setattr(label_studio_module.urllib_request, "urlopen", fake_urlopen)

    client = _LabelStudioHttpClient(
        base_url="http://label-studio.local",
        api_token="token",
        max_retries=2,
        retry_backoff_seconds=0,
    )

    with pytest.raises(_LabelStudioApiError, match="status=500"):
        client.request("POST", "/api/tasks/", payload={"data": {"text": "sample"}})

    assert len(calls) == 1


def test_poll_round_does_not_count_unparseable_label_studio_annotations_as_ready() -> None:
    backend = LabelStudioBackend(_external_config())
    backend._ready = True
    backend._project_id = "10"
    backend._project_ref = {"backend": "label_studio", "project_id": "10"}
    backend._label_schema = LabelSchema(task="text_classification", labels=["positive", "negative"])
    backend._http_client = _StaticTaskClient(
        {
            "id": 100,
            "is_labeled": True,
            "meta": {"sdk_round_id": "r1", "sdk_sample_id": "s1"},
            "annotations": [
                {
                    "completed_by": {"id": 7},
                    "created_at": "2026-04-28T10:00:00Z",
                    "result": [{"from_name": "label", "to_name": "text", "type": "choices", "value": {}}],
                }
            ],
        }
    )

    progress = backend.poll_round("r1", {"s1": "100"}, AnnotationPolicy(min_votes=1))

    assert progress.done == 0
    assert progress.ready_sample_ids == []
    assert progress.details["tasks"]["s1"]["eligible_votes"] == 0


def test_poll_round_does_not_count_schema_invalid_label_studio_annotations_as_ready() -> None:
    backend = LabelStudioBackend(_external_config())
    backend._ready = True
    backend._project_id = "10"
    backend._project_ref = {"backend": "label_studio", "project_id": "10"}
    backend._label_schema = LabelSchema(task="text_classification", labels=["positive", "negative"])
    backend._http_client = _StaticTaskClient(
        {
            "id": 100,
            "is_labeled": True,
            "meta": {"sdk_round_id": "r1", "sdk_sample_id": "s1"},
            "annotations": [
                {
                    "completed_by": {"id": 7},
                    "created_at": "2026-04-28T10:00:00Z",
                    "result": [
                        {
                            "from_name": "label",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": ["outside-schema"]},
                        }
                    ],
                }
            ],
        }
    )

    progress = backend.poll_round("r1", {"s1": "100"}, AnnotationPolicy(min_votes=1))

    assert progress.done == 0
    assert progress.ready_sample_ids == []
    assert progress.details["tasks"]["s1"]["eligible_votes"] == 0


def test_managed_runtime_requires_explicit_token_or_secret_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACTIVE_LEARNING_LABEL_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv("ACTIVE_LEARNING_LABEL_STUDIO_USERNAME", raising=False)
    monkeypatch.delenv("ACTIVE_LEARNING_LABEL_STUDIO_PASSWORD", raising=False)
    monkeypatch.delenv("ACTIVE_LEARNING_LABEL_STUDIO_TOKEN", raising=False)

    with pytest.raises(InfrastructureError, match="token|credential|secret"):
        ManagedLabelStudioRuntime(_managed_config(api_token=None))


def test_packaged_managed_label_studio_proxy_binds_loopback_only() -> None:
    packaged_root = resources.files("active_learning_sdk.backends.assets").joinpath("label_studio")
    compose_text = packaged_root.joinpath("docker-compose.yml").read_text(encoding="utf-8")

    assert "127.0.0.1:${LABEL_STUDIO_HOST_PORT:-8080}:8080" in compose_text


def test_fallback_managed_label_studio_proxy_binds_loopback_only() -> None:
    compose_text = (Path("docker") / "label_studio" / "docker-compose.yml").read_text(encoding="utf-8")

    assert "127.0.0.1:${LABEL_STUDIO_HOST_PORT:-8080}:8080" in compose_text


def test_compose_command_wraps_docker_version_timeout_as_infrastructure_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    runtime = ManagedLabelStudioRuntime(_managed_config(managed_compose_path=str(compose_file)))

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 10))

    monkeypatch.setattr(managed_docker_module.subprocess, "run", fake_run)

    with pytest.raises(InfrastructureError, match="Docker Compose|timed out"):
        runtime.compose_command()
