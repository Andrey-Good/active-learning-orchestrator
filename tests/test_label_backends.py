from __future__ import annotations

import io
import json
import math
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error

import pytest

from active_learning_sdk.backends import label_studio as label_studio_module
from active_learning_sdk.backends.label_studio import (
    _LabelStudioApiError,
    _LabelStudioHttpClient,
    LabelStudioBackend,
)
from active_learning_sdk.backends.simulator import SimulatorLabelBackend
from active_learning_sdk.configs import AnnotationPolicy, LabelBackendConfig, LabelSchema
from active_learning_sdk.exceptions import ConfigurationError, LabelBackendError
from active_learning_sdk.types import DataSample


def _schema(*, multi_label: bool = False) -> LabelSchema:
    return LabelSchema(task="text_classification", labels=["positive", "negative"], multi_label=multi_label)


def _label_studio_config(*, project_id: int | None = None) -> LabelBackendConfig:
    return LabelBackendConfig(
        backend="label_studio",
        mode="external",
        url="http://label-studio.local",
        api_token="token",
        project_id=project_id,
    )


def test_simulator_lifecycle_idempotency_min_votes_and_sorted_pull() -> None:
    backend = SimulatorLabelBackend()
    samples = [
        DataSample(sample_id="b", data={"text": "second"}),
        DataSample(sample_id="a", data={"text": "first"}),
    ]

    with pytest.raises(LabelBackendError):
        backend.push_round("r1", samples)
    with pytest.raises(LabelBackendError):
        backend.poll_round("r1", {}, AnnotationPolicy(min_votes=1))
    with pytest.raises(LabelBackendError):
        backend.pull_round("r1", {})

    assert backend.ensure_ready(_schema()) == {"backend": "simulator"}

    first_push = backend.push_round("r1", samples)
    second_push = backend.push_round("r1", list(reversed(samples)))
    assert first_push.task_ids == {"b": "sim:r1:b", "a": "sim:r1:a"}
    assert second_push.task_ids == {"a": "sim:r1:a", "b": "sim:r1:b"}

    backend.submit_annotation(round_id="r1", sample_id="a", value="positive", annotator_id="late", created_at=20.0)
    backend.submit_annotation(round_id="r1", sample_id="a", value="negative", annotator_id="early", created_at=10.0)
    backend.submit_annotation(round_id="r1", sample_id="b", value="negative", created_at=30.0)

    progress = backend.poll_round("r1", first_push.task_ids, AnnotationPolicy(min_votes=2))
    assert progress.total == 2
    assert progress.done == 1
    assert progress.ready_sample_ids == ["a"]

    pulled = backend.pull_round("r1", first_push.task_ids)
    assert [record.annotator_id for record in pulled.annotations["a"]] == ["early", "late"]
    assert [record.value for record in pulled.annotations["b"]] == ["negative"]


def test_simulator_oracle_modes_preserve_manual_annotations() -> None:
    push_oracle = SimulatorLabelBackend(label_by_sample_id={"s1": "positive"})
    push_oracle.ensure_ready(_schema())
    push_result = push_oracle.push_round("r1", [DataSample(sample_id="s1", data={"text": "hello"})])
    assert push_oracle.poll_round("r1", push_result.task_ids, AnnotationPolicy(min_votes=1)).done == 1
    assert push_oracle.pull_round("r1", push_result.task_ids).annotations["s1"][0].value == "positive"

    pull_oracle = SimulatorLabelBackend(label_fn=lambda sample: f"label:{sample.sample_id}", oracle_on="pull")
    pull_oracle.ensure_ready(_schema())
    pull_result = pull_oracle.push_round("r2", [DataSample(sample_id="s2", data={"text": "world"})])
    assert pull_oracle.poll_round("r2", pull_result.task_ids, AnnotationPolicy(min_votes=1)).done == 0
    assert pull_oracle.pull_round("r2", pull_result.task_ids).annotations["s2"][0].value == "label:s2"

    pull_oracle.submit_annotation(round_id="r2", sample_id="s2", value="manual", created_at=5.0)
    assert [record.value for record in pull_oracle.pull_round("r2", pull_result.task_ids).annotations["s2"]] == [
        "label:s2",
        "manual",
    ]


class _ProjectClient:
    def __init__(self, *, existing_project: Mapping[str, Any] | None = None) -> None:
        self.existing_project = dict(existing_project) if existing_project is not None else None
        self.calls: list[tuple[str, str, Mapping[str, Any] | None, Any]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> Any:
        self.calls.append((method, path, query, payload))
        if method == "GET" and path == "/api/projects/" and query and query.get("page_size") == 1:
            return {"results": [], "next": None}
        if method == "GET" and path == "/api/projects/" and query and query.get("title"):
            results = [self.existing_project] if self.existing_project is not None else []
            return {"results": results, "next": None}
        if method == "GET" and path == "/api/projects/7/":
            assert self.existing_project is not None
            return dict(self.existing_project)
        if method == "POST" and path == "/api/projects/":
            return {"id": 42, "title": payload["title"], "label_config": payload["label_config"]}
        if method == "PATCH" and path == "/api/projects/7/":
            patched = dict(self.existing_project or {})
            patched.update(payload)
            patched["id"] = 7
            return patched
        raise AssertionError(f"unexpected request: {method} {path} query={query} payload={payload}")


def test_label_studio_project_creation_and_existing_config_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    create_client = _ProjectClient()
    monkeypatch.setattr(label_studio_module, "_LabelStudioHttpClient", lambda **_: create_client)

    created_backend = LabelStudioBackend(_label_studio_config())
    created_ref = created_backend.ensure_ready(_schema())

    assert created_ref["project_id"] == "42"
    assert any(call[0] == "POST" and call[1] == "/api/projects/" for call in create_client.calls)

    compatible_backend = LabelStudioBackend(_label_studio_config(project_id=7))
    desired_config = compatible_backend._build_label_config(_schema())
    compatible_client = _ProjectClient(
        existing_project={"id": 7, "title": "Existing", "label_config": desired_config}
    )
    monkeypatch.setattr(label_studio_module, "_LabelStudioHttpClient", lambda **_: compatible_client)

    reused_ref = compatible_backend.ensure_ready(_schema())

    patch_calls = [call for call in compatible_client.calls if call[0] == "PATCH"]
    assert reused_ref["project_id"] == "7"
    assert patch_calls == []


def test_label_studio_reused_project_rejects_incompatible_config(monkeypatch: pytest.MonkeyPatch) -> None:
    explicit_client = _ProjectClient(existing_project={"id": 7, "title": "Existing", "label_config": "<old />"})
    monkeypatch.setattr(label_studio_module, "_LabelStudioHttpClient", lambda **_: explicit_client)

    with pytest.raises(ConfigurationError, match="incompatible label_config"):
        LabelStudioBackend(_label_studio_config(project_id=7)).ensure_ready(_schema())

    explicit_patch_calls = [call for call in explicit_client.calls if call[0] == "PATCH"]
    assert explicit_patch_calls == []

    search_backend = LabelStudioBackend(_label_studio_config())
    title = search_backend._project_title(_schema())
    title_client = _ProjectClient(existing_project={"id": 8, "title": title, "label_config": "<old />"})
    monkeypatch.setattr(label_studio_module, "_LabelStudioHttpClient", lambda **_: title_client)

    with pytest.raises(ConfigurationError, match="incompatible label_config"):
        search_backend.ensure_ready(_schema())

    title_patch_calls = [call for call in title_client.calls if call[0] == "PATCH"]
    assert title_patch_calls == []


class _TaskClient:
    def __init__(self) -> None:
        matching_prediction = {
            "model_version": "active-learning-sdk",
            "result": [
                {
                    "from_name": "label",
                    "to_name": "text",
                    "type": "choices",
                    "value": {"choices": ["positive"]},
                }
            ],
        }
        self.tasks: dict[str, dict[str, Any]] = {
            "100": {
                "id": 100,
                "meta": {"sdk_round_id": "r1", "sdk_sample_id": "s1"},
                "predictions": [matching_prediction],
                "annotations": [],
            }
        }
        self.calls: list[tuple[str, str, Mapping[str, Any] | None, Any]] = []
        self.prediction_imports: list[list[dict[str, Any]]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> Any:
        self.calls.append((method, path, query, payload))
        if method == "GET" and path == "/api/tasks/":
            return {"results": list(self.tasks.values()), "next": None}
        if method == "POST" and path == "/api/tasks/":
            task_id = "101"
            task = {"id": 101, "meta": dict(payload["meta"]), "predictions": [], "annotations": []}
            self.tasks[task_id] = task
            return task
        if method == "GET" and path.startswith("/api/tasks/"):
            task_id = path.strip("/").split("/")[-1]
            return self.tasks[task_id]
        if method == "POST" and path == "/api/projects/10/import/predictions":
            self.prediction_imports.append(payload)
            for prediction in payload:
                self.tasks[str(prediction["task"])]["predictions"].append(
                    {"model_version": prediction["model_version"], "result": prediction["result"]}
                )
            return {"created": len(payload)}
        raise AssertionError(f"unexpected request: {method} {path} query={query} payload={payload}")


def test_label_studio_task_idempotency_prediction_import_and_duplicate_avoidance() -> None:
    backend = LabelStudioBackend(_label_studio_config(project_id=10))
    client = _TaskClient()
    backend._ready = True
    backend._project_id = "10"
    backend._project_ref = {"backend": "label_studio", "project_id": "10"}
    backend._label_schema = _schema()
    backend._http_client = client

    samples = [
        DataSample(sample_id="s1", data={"text": "already exists"}),
        DataSample(sample_id="s2", data={"text": "new task"}),
    ]

    first = backend.push_round("r1", samples, prelabels={"s1": "positive", "s2": "negative"})
    second = backend.push_round("r1", samples, prelabels={"s1": "positive", "s2": "negative"})

    assert first.task_ids == {"s1": "100", "s2": "101"}
    assert second.task_ids == first.task_ids
    assert len([call for call in client.calls if call[0] == "POST" and call[1] == "/api/tasks/"]) == 1
    assert client.prediction_imports == [
        [
            {
                "task": 101,
                "model_version": "active-learning-sdk",
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "text",
                        "type": "choices",
                        "value": {"choices": ["negative"]},
                    }
                ],
            }
        ]
    ]


class _AmbiguousCreateClient(_TaskClient):
    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> Any:
        if method == "GET" and path == "/api/tasks/":
            self.calls.append((method, path, query, payload))
            return {"results": list(self.tasks.values()), "next": None}
        if method == "POST" and path == "/api/tasks/":
            self.calls.append((method, path, query, payload))
            task = {"id": 101, "meta": dict(payload["meta"]), "predictions": [], "annotations": []}
            self.tasks["101"] = task
            raise _LabelStudioApiError("connection reset after create")
        return super().request(method, path, query=query, payload=payload)


def test_label_studio_push_recovers_task_id_after_ambiguous_create_failure() -> None:
    backend = LabelStudioBackend(_label_studio_config(project_id=10))
    client = _AmbiguousCreateClient()
    client.tasks = {}
    backend._ready = True
    backend._project_id = "10"
    backend._project_ref = {"backend": "label_studio", "project_id": "10"}
    backend._label_schema = _schema()
    backend._http_client = client

    pushed = backend.push_round("r1", [DataSample(sample_id="s1", data={"text": "created then failed"})])

    assert pushed.task_ids == {"s1": "101"}
    assert len([call for call in client.calls if call[0] == "POST" and call[1] == "/api/tasks/"]) == 1


class _AmbiguousPredictionImportClient(_TaskClient):
    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> Any:
        if method == "POST" and path == "/api/projects/10/import/predictions":
            self.calls.append((method, path, query, payload))
            self.prediction_imports.append(payload)
            for prediction in payload:
                self.tasks[str(prediction["task"])]["predictions"].append(
                    {"model_version": prediction["model_version"], "result": prediction["result"]}
                )
            raise _LabelStudioApiError("connection reset after prediction import")
        return super().request(method, path, query=query, payload=payload)


def test_label_studio_push_recovers_after_ambiguous_prediction_import_failure() -> None:
    backend = LabelStudioBackend(_label_studio_config(project_id=10))
    client = _AmbiguousPredictionImportClient()
    client.tasks = {}
    backend._ready = True
    backend._project_id = "10"
    backend._project_ref = {"backend": "label_studio", "project_id": "10"}
    backend._label_schema = _schema()
    backend._http_client = client

    pushed = backend.push_round(
        "r1",
        [DataSample(sample_id="s1", data={"text": "prelabeled"})],
        prelabels={"s1": "positive"},
    )

    assert pushed.task_ids == {"s1": "101"}
    assert len(client.prediction_imports) == 1


def test_label_studio_push_reports_unrecoverable_ambiguous_create_failure() -> None:
    backend = LabelStudioBackend(_label_studio_config(project_id=10))
    client = _AmbiguousCreateClient()
    client.tasks = {}
    backend._ready = True
    backend._project_id = "10"
    backend._project_ref = {"backend": "label_studio", "project_id": "10"}
    backend._label_schema = _schema()
    backend._http_client = client

    def empty_find(*, round_id: str, sample_ids: Sequence[str]) -> dict[str, str]:
        return {}

    backend._find_existing_tasks = empty_find  # type: ignore[method-assign]

    with pytest.raises(LabelBackendError, match="could not recover all task ids"):
        backend.push_round("r1", [DataSample(sample_id="s1", data={"text": "lost"})])


def test_label_studio_malformed_success_payloads_raise_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    create_client = _ProjectClient()
    original_request = create_client.request

    def malformed_project_request(
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        payload: Any = None,
    ) -> Any:
        if method == "GET" and path == "/api/projects/" and query and query.get("page_size") == 1:
            return {"results": [], "next": None}
        if method == "GET" and path == "/api/projects/" and query and query.get("title"):
            return {"results": [], "next": None}
        if method == "POST" and path == "/api/projects/":
            return {"title": payload["title"], "label_config": payload["label_config"]}
        return original_request(method, path, query=query, payload=payload)

    create_client.request = malformed_project_request  # type: ignore[method-assign]
    monkeypatch.setattr(label_studio_module, "_LabelStudioHttpClient", lambda **_: create_client)

    with pytest.raises(LabelBackendError, match="Malformed Label Studio API response.*id"):
        LabelStudioBackend(_label_studio_config()).ensure_ready(_schema())

    backend = LabelStudioBackend(_label_studio_config(project_id=10))
    backend._ready = True
    backend._project_id = "10"
    backend._project_ref = {"backend": "label_studio", "project_id": "10"}
    backend._label_schema = _schema()
    backend._http_client = _TaskClient()

    backend._http_client.tasks["100"]["annotations"] = "not-a-list"  # type: ignore[union-attr]
    with pytest.raises(LabelBackendError, match="annotations"):
        backend.poll_round("r1", {"s1": "100"}, AnnotationPolicy(min_votes=1))

    backend._http_client = _TaskClient()
    backend._http_client.tasks["100"]["meta"] = "not-a-mapping"  # type: ignore[union-attr]
    with pytest.raises(LabelBackendError, match="meta.*mapping"):
        backend.poll_round("r1", {"s1": "100"}, AnnotationPolicy(min_votes=1))

    backend._http_client = _TaskClient()
    backend._http_client.tasks["100"]["meta"] = "not-a-mapping"  # type: ignore[union-attr]
    with pytest.raises(LabelBackendError, match="meta.*mapping"):
        backend.push_round("r1", [DataSample(sample_id="s1", data={"text": "already exists"})])

    class BadGetTaskClient:
        def request(
            self,
            method: str,
            path: str,
            *,
            query: Mapping[str, Any] | None = None,
            payload: Any = None,
        ) -> Any:
            return []

    backend._http_client = BadGetTaskClient()  # type: ignore[assignment]
    with pytest.raises(LabelBackendError, match="mapping"):
        backend._get_task("missing")


def test_label_studio_annotation_parsing_and_non_finite_scores_are_safe() -> None:
    single_backend = LabelStudioBackend(_label_studio_config(project_id=10))
    single_backend._label_schema = _schema()

    single_record = single_backend._parse_annotation(
        {
            "completed_by": {"email": "annotator@example.com"},
            "created_at": "2026-04-25T10:00:00Z",
            "score": "0.75",
            "result": [{"value": {"choices": ["positive"]}}],
        }
    )
    assert single_record is not None
    assert single_record.annotator_id == "annotator@example.com"
    assert single_record.value == "positive"
    assert single_record.score == 0.75

    unsafe_score_record = single_backend._parse_annotation(
        {
            "completed_by": 123,
            "score": float("inf"),
            "result": [{"value": {"choices": ["negative"]}}],
        }
    )
    assert unsafe_score_record is not None
    assert unsafe_score_record.score is None

    multi_backend = LabelStudioBackend(_label_studio_config(project_id=10))
    multi_backend._label_schema = _schema(multi_label=True)
    multi_record = multi_backend._parse_annotation(
        {
            "created_username": "multi",
            "score": float("nan"),
            "result": [{"value": {"choices": ["positive", "negative"]}}],
        }
    )
    assert multi_record is not None
    assert multi_record.value == ["positive", "negative"]
    assert multi_record.score is None

    with pytest.raises(LabelBackendError, match="finite numeric values"):
        single_backend._build_prediction({"positive": float("nan"), "negative": 0.25})

    with pytest.raises(LabelBackendError, match="finite numeric probabilities"):
        single_backend._build_prediction([float("nan"), float("inf")])
    probability_prediction = single_backend._build_prediction([0.75, 0.25])
    assert probability_prediction is not None
    assert probability_prediction["score"] == 0.75
    assert probability_prediction["result"][0]["value"]["choices"] == ["positive"]
    assert single_backend._build_prediction_import(task_id="1", prelabel="positive") == {
        "task": 1,
        "model_version": "active-learning-sdk",
        "result": [
            {
                "from_name": "label",
                "to_name": "text",
                "type": "choices",
                "value": {"choices": ["positive"]},
            }
        ],
    }


def test_label_studio_direct_probability_prelabels_reject_non_unit_rows() -> None:
    backend = LabelStudioBackend(_label_studio_config(project_id=10))
    client = _TaskClient()
    backend._ready = True
    backend._project_id = "10"
    backend._project_ref = {"backend": "label_studio", "project_id": "10"}
    backend._label_schema = _schema()
    backend._http_client = client

    with pytest.raises(LabelBackendError, match="sum to 1.0"):
        backend.push_round(
            "r1",
            [DataSample(sample_id="s2", data={"text": "new task"})],
            prelabels={"s2": [0.6, 0.6]},
        )

    assert client.prediction_imports == []


def test_simulator_manual_non_finite_scores_are_normalized() -> None:
    backend = SimulatorLabelBackend()
    backend.ensure_ready(_schema())
    push_result = backend.push_round("r1", [DataSample(sample_id="s1", data={"text": "hello"})])

    backend.submit_annotation(round_id="r1", sample_id="s1", value="positive", score=float("nan"))
    backend.submit_annotation(round_id="r1", sample_id="s1", value="positive", score=float("inf"))
    backend.submit_annotation(round_id="r1", sample_id="s1", value="positive", score=0.75)

    annotations = backend.pull_round("r1", push_result.task_ids).annotations["s1"]
    assert [record.score for record in annotations] == [None, None, 0.75]
    json.dumps([record.__dict__ for record in annotations], allow_nan=False)


class _FakeUrlResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeUrlResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


@pytest.mark.parametrize("status_code", [408, 429, 500])
def test_label_studio_http_client_retries_transient_http_statuses(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def fake_urlopen(request: Any, timeout: int) -> _FakeUrlResponse:
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib_error.HTTPError(
                request.full_url,
                status_code,
                "temporary",
                {},
                io.BytesIO(b"temporary body"),
            )
        return _FakeUrlResponse(b'{"ok": true}')

    monkeypatch.setattr(label_studio_module.urllib_request, "urlopen", fake_urlopen)

    client = _LabelStudioHttpClient(
        base_url="http://label-studio.local",
        api_token="token",
        max_retries=2,
        retry_backoff_seconds=0.5,
        sleep_fn=sleeps.append,
    )

    assert client.request("GET", "/api/projects/") == {"ok": True}
    assert len(calls) == 2
    assert sleeps == [0.5]


def test_label_studio_http_client_retries_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request: Any, timeout: int) -> _FakeUrlResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib_error.URLError("connection reset")
        return _FakeUrlResponse(b'{"ok": true}')

    monkeypatch.setattr(label_studio_module.urllib_request, "urlopen", fake_urlopen)

    client = _LabelStudioHttpClient(
        base_url="http://label-studio.local",
        api_token="token",
        max_retries=2,
        retry_backoff_seconds=0.25,
        sleep_fn=sleeps.append,
    )

    assert client.request("GET", "/api/projects/") == {"ok": True}
    assert calls == 2
    assert sleeps == [0.25]


def test_label_studio_http_client_does_not_retry_permanent_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_urlopen(request: Any, timeout: int) -> _FakeUrlResponse:
        nonlocal calls
        calls += 1
        raise urllib_error.HTTPError(request.full_url, 400, "bad request", {}, io.BytesIO(b"bad request body"))

    monkeypatch.setattr(label_studio_module.urllib_request, "urlopen", fake_urlopen)

    client = _LabelStudioHttpClient(base_url="http://label-studio.local", api_token="token", max_retries=3)

    with pytest.raises(_LabelStudioApiError) as error:
        client.request("GET", "/api/projects/")

    assert calls == 1
    assert error.value.status_code == 400
    assert "GET http://label-studio.local/api/projects/ status=400 body=bad request body" in str(error.value)


def test_label_studio_http_client_rejects_non_strict_json_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_urlopen(request: Any, timeout: int) -> _FakeUrlResponse:
        nonlocal calls
        calls += 1
        return _FakeUrlResponse(b"{}")

    monkeypatch.setattr(label_studio_module.urllib_request, "urlopen", fake_urlopen)

    client = _LabelStudioHttpClient(base_url="http://label-studio.local", api_token="token")

    with pytest.raises(_LabelStudioApiError) as error:
        client.request("POST", "/api/tasks/", payload={"score": math.nan})

    assert calls == 0
    assert "payload is not strict JSON-safe" in str(error.value)


def test_label_studio_http_client_wraps_non_serializable_payload_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request: Any, timeout: int) -> _FakeUrlResponse:
        nonlocal calls
        calls += 1
        return _FakeUrlResponse(b"{}")

    monkeypatch.setattr(label_studio_module.urllib_request, "urlopen", fake_urlopen)

    client = _LabelStudioHttpClient(base_url="http://label-studio.local", api_token="token")

    with pytest.raises(_LabelStudioApiError) as error:
        client.request("POST", "/api/tasks/", payload={"created_at": object()})

    assert calls == 0
    assert "payload is not strict JSON-safe" in str(error.value)
