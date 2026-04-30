"""
Label Studio backend implementation.
"""

from __future__ import annotations


import hashlib
import html
import json
import math
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from ..configs import AnnotationPolicy, LabelBackendConfig, LabelSchema
from ..exceptions import ConfigurationError, LabelBackendError
from ..types import AnnotationRecord, DataSample
from .base import RoundProgress, RoundPullResult, RoundPushResult
from .managed_docker import ManagedLabelStudioRuntime

_META_ROUND_ID = "sdk_round_id"
_META_SAMPLE_ID = "sdk_sample_id"
_META_EXTERNAL_ID = "sdk_external_id"
_POLL_TIMEOUT_SECONDS = 120
_CONTROL_NAME = "label"
_OBJECT_NAME = "text"


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _auth_header_value(token: str) -> str:
    if token.startswith("Token ") or token.startswith("Bearer "):
        return token
    return f"Token {token}"


def _parse_timestamp(value: Any) -> float:
    if value is None:
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            return time.time()
    return time.time()


class _LabelStudioHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_token: str,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.25,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.api_token = api_token
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._sleep_fn = sleep_fn

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Mapping[str, Any]] = None,
        payload: Optional[Any] = None,
    ) -> Any:
        url = self._build_url(path, query=query)
        body: Optional[bytes] = None
        headers = {"Authorization": _auth_header_value(self.api_token), "Accept": "application/json"}
        if payload is not None:
            try:
                body = json.dumps(payload, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as error:
                raise _LabelStudioApiError(
                    f"Label Studio API request payload is not strict JSON-safe: {method.upper()} {url} error={error}"
                ) from error
            headers["Content-Type"] = "application/json"

        method_upper = method.upper()
        for attempt in range(self.max_retries + 1):
            request = urllib_request.Request(url=url, data=body, headers=headers, method=method_upper)
            try:
                with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                    if not raw:
                        return {}
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return {"raw": raw}
            except urllib_error.HTTPError as error:
                message = error.read().decode("utf-8", errors="replace")
                if self._should_retry_http_status(method_upper, error.code, attempt):
                    self._sleep_before_retry(attempt)
                    continue
                raise _LabelStudioApiError(
                    f"Label Studio API request failed: {method_upper} {url} status={error.code} body={message}",
                    status_code=error.code,
                ) from error
            except urllib_error.URLError as error:
                if self._should_retry_transport_error(method_upper, attempt):
                    self._sleep_before_retry(attempt)
                    continue
                raise _LabelStudioApiError(
                    f"Failed to reach Label Studio: {method_upper} {url} error={error.reason}"
                ) from error

        raise _LabelStudioApiError(f"Label Studio API request failed after retries: {method_upper} {url}")

    def _build_url(self, path: str, *, query: Optional[Mapping[str, Any]] = None) -> str:
        absolute = urllib_parse.urljoin(f"{self.base_url}/", path.lstrip("/"))
        if not query:
            return absolute
        filtered = {key: value for key, value in query.items() if value is not None}
        return f"{absolute}?{urllib_parse.urlencode(filtered)}"

    def _should_retry_http_status(self, method: str, status_code: int, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        if not self._is_retry_safe_method(method):
            return False
        return status_code in {408, 429} or 500 <= status_code <= 599

    def _should_retry_transport_error(self, method: str, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        return self._is_retry_safe_method(method)

    def _is_retry_safe_method(self, method: str) -> bool:
        return method.upper() in {"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"}

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        if delay > 0:
            self._sleep_fn(delay)


class _LabelStudioApiError(LabelBackendError):
    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LabelStudioBackend:
    """
    Label Studio backend with external and managed-docker modes.
    """

    def __init__(self, config: LabelBackendConfig) -> None:
        config.validate()
        self.config = config
        self._ready = False
        self._project_ref: Dict[str, Any] = {}
        self._http_client: Optional[_LabelStudioHttpClient] = None
        self._runtime = ManagedLabelStudioRuntime(config) if config.mode == "managed_docker" else None
        self._project_id: Optional[str] = str(config.project_id) if config.project_id is not None else None
        self._label_schema: Optional[LabelSchema] = None
        self._resolved_url = self._resolve_url()
        self._resolved_token = self._resolve_token()

    def ensure_ready(self, label_schema: LabelSchema) -> Dict[str, Any]:
        label_schema.validate()
        self._label_schema = label_schema
        try:
            if self._runtime is not None:
                self._runtime.ensure_running()
            self._http_client = _LabelStudioHttpClient(base_url=self._resolved_url, api_token=self._resolved_token)
            project = self._ensure_project(label_schema)
            self._project_id = self._require_id(project, endpoint="/api/projects/", context="ensure project")
            self._project_ref = {
                "backend": "label_studio",
                "mode": self.config.mode,
                "project_id": self._project_id,
                "url": self._resolved_url,
                "project_title": project.get("title"),
            }
            self._ready = True
            return dict(self._project_ref)
        except Exception:
            self._ready = False
            self._http_client = None
            self._project_ref = {}
            raise

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: Optional[Dict[str, Any]] = None,
    ) -> RoundPushResult:
        self._require_ready()
        sample_ids = [sample.sample_id for sample in samples]
        existing = self._find_existing_tasks(round_id=round_id, sample_ids=sample_ids)
        task_ids: Dict[str, str] = dict(existing)
        for sample in samples:
            if sample.sample_id in task_ids:
                continue
            payload = self._build_task_payload(round_id=round_id, sample=sample)
            try:
                task = self._client().request("POST", "/api/tasks/", payload=payload)
            except LabelBackendError as error:
                task_ids = self._recover_task_ids_after_push_failure(
                    round_id=round_id,
                    sample_ids=sample_ids,
                    required_sample_ids=[sample.sample_id],
                    original_error=error,
                    context="task create",
                )
                continue
            task_id = self._require_id(task, endpoint="/api/tasks/", context=f"create task sample_id={sample.sample_id!r}")
            task_ids[sample.sample_id] = task_id

        prediction_imports = self._collect_missing_prediction_imports(samples=samples, task_ids=task_ids, prelabels=prelabels)

        if prediction_imports:
            endpoint = f"/api/projects/{self._project_id}/import/predictions"
            try:
                self._client().request("POST", endpoint, payload=prediction_imports)
            except LabelBackendError as error:
                missing_after_failure = self._collect_missing_prediction_imports(
                    samples=samples,
                    task_ids=task_ids,
                    prelabels=prelabels,
                )
                if missing_after_failure:
                    raise LabelBackendError(
                        "Label Studio prediction import failed and metadata reconciliation "
                        f"could not confirm import completion: endpoint={endpoint!r}, "
                        f"missing_prediction_count={len(missing_after_failure)}, original_error={error}"
                    ) from error

        return RoundPushResult(task_ids=task_ids, backend_round_ref=dict(self._project_ref))

    def recover_push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> RoundPushResult:
        self._require_ready()
        sample_ids = [sample.sample_id for sample in samples]
        task_ids = self._recover_task_ids_after_push_failure(
            round_id=round_id,
            sample_ids=sample_ids,
            original_error=error,
            context="engine push recovery",
        )
        missing_prediction_imports = self._collect_missing_prediction_imports(
            samples=samples,
            task_ids=task_ids,
            prelabels=prelabels,
        )
        if missing_prediction_imports:
            raise LabelBackendError(
                "Label Studio push recovery found all task ids but could not confirm "
                f"{len(missing_prediction_imports)} prediction imports without unsafe POST retry."
            )
        return RoundPushResult(task_ids=task_ids, backend_round_ref=dict(self._project_ref))

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        self._require_ready()
        ready_sample_ids: List[str] = []
        details: Dict[str, Any] = {}
        for sample_id, task_id in task_ids.items():
            task = self._get_task(task_id)
            self._validate_task_binding(task, round_id=round_id, sample_id=sample_id, task_id=str(task_id))
            annotations = self._extract_annotations(task)
            ready_votes = self._ready_vote_count(annotations, policy)
            if ready_votes >= policy.min_votes:
                ready_sample_ids.append(sample_id)
            details[sample_id] = {
                "task_id": task_id,
                "annotations": len(annotations),
                "eligible_votes": ready_votes,
                "is_labeled": task.get("is_labeled"),
                "round_id": round_id,
            }

        return RoundProgress(
            total=len(task_ids),
            done=len(ready_sample_ids),
            ready_sample_ids=ready_sample_ids,
            details={"project_id": self._project_id, "tasks": details},
        )

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        self._require_ready()
        annotations: Dict[str, List[AnnotationRecord]] = {}
        backend_payload: Dict[str, Any] = {"project_id": self._project_id, "round_id": round_id, "tasks": {}}

        for sample_id, task_id in task_ids.items():
            task = self._get_task(task_id)
            self._validate_task_binding(task, round_id=round_id, sample_id=sample_id, task_id=str(task_id))
            parsed = [record for record in (self._parse_annotation(annotation) for annotation in self._extract_annotations(task)) if record is not None]
            parsed.sort(key=lambda record: record.created_at)
            annotations[sample_id] = parsed
            backend_payload["tasks"][sample_id] = {"task_id": task_id, "annotation_count": len(parsed)}

        return RoundPullResult(annotations=annotations, backend_payload=backend_payload)

    def close(self) -> None:
        self._ready = False
        self._project_ref = {}
        self._http_client = None
        self._project_id = None
        self._label_schema = None

    def _client(self) -> _LabelStudioHttpClient:
        if self._http_client is None:
            raise LabelBackendError("Label Studio HTTP client is not initialized.")
        return self._http_client

    def _require_ready(self) -> None:
        if not self._ready:
            raise LabelBackendError("LabelStudioBackend is not ready. Call ensure_ready() first.")

    def _resolve_url(self) -> str:
        if self.config.mode == "managed_docker":
            if self._runtime is None:
                raise ConfigurationError("Managed Docker runtime was not initialized.")
            return self._runtime.resolved_url
        assert self.config.url is not None
        return _normalize_base_url(self.config.url)

    def _resolve_token(self) -> str:
        if self.config.mode == "managed_docker" and self._runtime is not None:
            return self._runtime.token
        if not self.config.api_token:
            raise ConfigurationError("Label Studio API token is required.")
        return self.config.api_token

    def _ensure_project(self, label_schema: LabelSchema) -> Dict[str, Any]:
        desired_label_config = self._build_label_config(label_schema)
        self._wait_until_api_ready()
        if self._project_id is not None:
            endpoint = f"/api/projects/{self._project_id}/"
            project = self._require_mapping(
                self._client().request("GET", endpoint),
                endpoint=endpoint,
                context="get configured project",
            )
            return self._ensure_project_config(project, desired_label_config)

        project_title = self._project_title(label_schema)
        for project in self._iter_projects(title=project_title):
            if project.get("title") == project_title:
                return self._ensure_project_config(project, desired_label_config)

        created = self._client().request(
            "POST",
            "/api/projects/",
            payload={
                "title": project_title,
                "description": "Managed by Active Learning SDK",
                "label_config": desired_label_config,
            },
        )
        return self._require_mapping(created, endpoint="/api/projects/", context="create project")

    def _ensure_project_config(self, project: Mapping[str, Any], desired_label_config: str) -> Dict[str, Any]:
        current = project.get("label_config")
        if current == desired_label_config:
            return dict(project)
        raise ConfigurationError(
            "Existing Label Studio project has an incompatible label_config. "
            f"Refusing to mutate project id={project.get('id')!r}, title={project.get('title')!r}; "
            "create a new project or update the project explicitly."
        )

    def _wait_until_api_ready(self) -> None:
        deadline = time.time() + _POLL_TIMEOUT_SECONDS
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                response = self._client().request("GET", "/api/projects/", query={"page_size": 1})
                self._extract_items_from_list_response(
                    response,
                    endpoint="/api/projects/",
                    context="readiness probe",
                    keys=("results",),
                )
                return
            except _LabelStudioApiError as error:
                if self._is_permanent_readiness_error(error):
                    raise
                last_error = error
                time.sleep(1.0)
        diagnostics = self._runtime.diagnostics() if self._runtime is not None else {}
        raise LabelBackendError(
            f"Label Studio did not become ready within {_POLL_TIMEOUT_SECONDS} seconds. "
            f"diagnostics={diagnostics}"
        ) from last_error

    def _is_permanent_readiness_error(self, error: _LabelStudioApiError) -> bool:
        if error.status_code is None:
            return False
        return 400 <= error.status_code < 500 and error.status_code not in {408, 429}

    def _build_label_config(self, label_schema: LabelSchema) -> str:
        if label_schema.task != "text_classification":
            raise ConfigurationError(
                f"LabelStudioBackend currently supports label_schema.task='text_classification', got {label_schema.task!r}."
            )

        label_lines = "\n".join(
            f'    <Choice value="{html.escape(label, quote=True)}" />' for label in label_schema.labels
        )
        choice_mode = "multiple" if label_schema.multi_label else "single"
        return (
            "<View>\n"
            "  <Header value=\"Choose label:\"/>\n"
            f"  <Text name=\"{_OBJECT_NAME}\" value=\"$text\"/>\n"
            f"  <Choices name=\"{_CONTROL_NAME}\" toName=\"{_OBJECT_NAME}\" choice=\"{choice_mode}\">\n"
            f"{label_lines}\n"
            "  </Choices>\n"
            "</View>"
        )

    def _project_title(self, label_schema: LabelSchema) -> str:
        fingerprint = hashlib.sha1(
            f"{label_schema.task}|{label_schema.multi_label}|{'|'.join(label_schema.labels)}".encode("utf-8")
        ).hexdigest()[:10]
        return f"Active Learning SDK [{label_schema.task}] {fingerprint}"

    def _iter_projects(self, *, title: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        page = 1
        while True:
            response = self._client().request(
                "GET",
                "/api/projects/",
                query={"page": page, "page_size": 100, "title": title},
            )
            projects = self._extract_items_from_list_response(
                response,
                endpoint="/api/projects/",
                context="list projects",
                keys=("results",),
            )
            if not projects:
                return
            for project in projects:
                yield dict(project)
            if not self._has_next_page(response):
                return
            page += 1

    def _find_existing_tasks(self, *, round_id: str, sample_ids: Sequence[str]) -> Dict[str, str]:
        wanted = set(sample_ids)
        found: Dict[str, str] = {}
        page = 1
        while wanted - set(found):
            response = self._client().request(
                "GET",
                "/api/tasks/",
                query={"project": self._project_id, "page": page, "page_size": 100},
            )
            tasks = self._extract_tasks_from_list_response(response, endpoint="/api/tasks/", context="find existing tasks")
            if not tasks:
                break
            for task in tasks:
                meta = self._task_meta(task, endpoint="/api/tasks/", context="find existing task metadata")
                sample_id = meta.get(_META_SAMPLE_ID)
                if meta.get(_META_ROUND_ID) == round_id and sample_id in wanted:
                    found[sample_id] = self._require_id(task, endpoint="/api/tasks/", context=f"listed task sample_id={sample_id!r}")
            if not self._has_next_page(response):
                break
            page += 1
        return found

    def _recover_task_ids_after_push_failure(
        self,
        *,
        round_id: str,
        sample_ids: Sequence[str],
        required_sample_ids: Optional[Sequence[str]] = None,
        original_error: Optional[Exception],
        context: str,
    ) -> Dict[str, str]:
        recovered = self._find_existing_tasks(round_id=round_id, sample_ids=sample_ids)
        required_ids = list(required_sample_ids) if required_sample_ids is not None else list(sample_ids)
        missing = [sample_id for sample_id in required_ids if sample_id not in recovered]
        if missing:
            raise LabelBackendError(
                "Label Studio push reconciliation could not recover all task ids after ambiguous failure: "
                f"context={context!r}, missing_sample_ids={missing}, original_error={original_error}"
            ) from original_error
        return recovered

    def _build_task_payload(self, *, round_id: str, sample: DataSample) -> Dict[str, Any]:
        if self._label_schema is None:
            raise LabelBackendError("Label schema is not initialized.")
        if self._label_schema.task == "text_classification" and "text" not in sample.data:
            raise ConfigurationError(f"Sample {sample.sample_id!r} is missing data['text'] required for text classification.")

        payload: Dict[str, Any] = {
            "project": int(self._project_id) if self._project_id is not None else None,
            "data": dict(sample.data),
            "meta": {
                **dict(sample.meta),
                _META_ROUND_ID: round_id,
                _META_SAMPLE_ID: sample.sample_id,
                _META_EXTERNAL_ID: f"{round_id}:{sample.sample_id}",
                "group_id": sample.group_id,
            },
        }
        return payload

    def _build_prediction_import(self, *, task_id: str, prelabel: Optional[Any]) -> Optional[Dict[str, Any]]:
        prediction = self._build_prediction(prelabel)
        if prediction is None:
            return None
        payload = {
            "task": int(task_id),
            "model_version": prediction["model_version"],
            "result": prediction["result"],
        }
        if prediction.get("score") is not None:
            payload["score"] = prediction["score"]
        return payload

    def _build_prediction(self, prelabel: Optional[Any]) -> Optional[Dict[str, Any]]:
        if prelabel is None or self._label_schema is None:
            return None

        label: Optional[Any] = None
        score: Optional[float] = None
        if isinstance(prelabel, str):
            label = prelabel
        elif isinstance(prelabel, Mapping):
            best_label: Optional[str] = None
            best_score = float("-inf")
            for key, value in prelabel.items():
                candidate_label = str(key)
                self._validate_prediction_choices([candidate_label])
                candidate_score = self._coerce_prediction_score(value)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_label = candidate_label
            label = best_label
            score = None if best_label is None else best_score
        elif isinstance(prelabel, Sequence) and not isinstance(prelabel, (bytes, bytearray)):
            values = list(prelabel)
            if values and all(isinstance(item, str) for item in values):
                self._validate_prediction_choices(values)
                if not self._label_schema.multi_label and len(values) != 1:
                    raise LabelBackendError("Single-label Label Studio prelabel must contain exactly one label.")
                label = values if self._label_schema.multi_label else values[0]
            elif values:
                numeric_scores = self._validate_probability_row(values)
                best_index = max(range(len(numeric_scores)), key=numeric_scores.__getitem__)
                best_score = numeric_scores[best_index]
                label = self._label_schema.labels[best_index]
                score = best_score

        if label is None:
            return None

        choices = list(label) if isinstance(label, list) else [label]
        self._validate_prediction_choices(choices)
        return {
            "model_version": "active-learning-sdk",
            "score": score,
            "result": [
                {
                    "from_name": _CONTROL_NAME,
                    "to_name": _OBJECT_NAME,
                    "type": "choices",
                    "value": {"choices": choices},
                }
            ],
        }

    def _get_task(self, task_id: str) -> Dict[str, Any]:
        endpoint = f"/api/tasks/{task_id}/"
        response = self._client().request("GET", endpoint)
        return self._require_mapping(response, endpoint=endpoint, context="get task")

    def _validate_task_binding(self, task: Mapping[str, Any], *, round_id: str, sample_id: str, task_id: str) -> None:
        meta = self._task_meta(task, endpoint="/api/tasks/<id>/", context=f"validate task binding task_id={task_id!r}")
        task_round_id = meta.get(_META_ROUND_ID)
        task_sample_id = meta.get(_META_SAMPLE_ID)
        if task_round_id != round_id:
            raise LabelBackendError(
                f"Label Studio task id {task_id!r} belongs to round_id={task_round_id!r}, not {round_id!r}."
            )
        if task_sample_id != sample_id:
            raise LabelBackendError(
                f"Label Studio task id {task_id!r} does not belong to sample_id={sample_id!r}."
            )

    def _collect_missing_prediction_imports(
        self,
        *,
        samples: Sequence[DataSample],
        task_ids: Mapping[str, str],
        prelabels: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not prelabels:
            return []

        imports: List[Dict[str, Any]] = []
        for sample in samples:
            if sample.sample_id not in prelabels:
                continue
            task_id = task_ids[sample.sample_id]
            prediction_import = self._build_prediction_import(task_id=task_id, prelabel=prelabels[sample.sample_id])
            if prediction_import is None:
                continue
            task = self._get_task(task_id)
            if self._task_has_matching_prediction(task, prediction_import):
                continue
            imports.append(prediction_import)
        return imports

    def _task_has_matching_prediction(self, task: Mapping[str, Any], prediction_import: Mapping[str, Any]) -> bool:
        expected_result = list(prediction_import.get("result") or [])
        expected_model_version = prediction_import.get("model_version")
        predictions = task.get("predictions", []) or []
        if not isinstance(predictions, list):
            raise LabelBackendError(
                "Malformed Label Studio task response: endpoint='/api/tasks/<id>/', "
                "context='check predictions', expected field 'predictions' to be a list."
            )
        for prediction in predictions:
            if not isinstance(prediction, Mapping):
                raise LabelBackendError(
                    "Malformed Label Studio task response: endpoint='/api/tasks/<id>/', "
                    "context='check predictions', expected prediction entries to be mappings."
                )
            if prediction.get("model_version") != expected_model_version:
                continue
            if list(prediction.get("result") or []) == expected_result:
                return True
        return False

    def _extract_items_from_list_response(
        self,
        response: Any,
        *,
        endpoint: str,
        context: str,
        keys: Sequence[str],
    ) -> List[Mapping[str, Any]]:
        items: List[Any]
        if isinstance(response, list):
            items = response
        elif isinstance(response, Mapping):
            matched_items: Optional[List[Any]] = None
            for key in keys:
                value = response.get(key)
                if isinstance(value, list):
                    matched_items = value
                    break
            if matched_items is None:
                raise self._malformed_response_error(
                    endpoint=endpoint,
                    context=context,
                    expected=f"list response or mapping with one of {list(keys)!r}",
                    response=response,
                )
            items = matched_items
        else:
            raise self._malformed_response_error(
                endpoint=endpoint,
                context=context,
                expected="list response or mapping",
                response=response,
            )
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise self._malformed_response_error(
                    endpoint=endpoint,
                    context=f"{context} item[{index}]",
                    expected="mapping item",
                    response=item,
                )
        return list(items)

    def _extract_tasks_from_list_response(self, response: Any, *, endpoint: str, context: str) -> List[Mapping[str, Any]]:
        return self._extract_items_from_list_response(
            response,
            endpoint=endpoint,
            context=context,
            keys=("tasks", "results"),
        )

    def _task_meta(self, task: Mapping[str, Any], *, endpoint: str, context: str) -> Dict[str, Any]:
        raw_meta = task.get("meta") or {}
        if isinstance(raw_meta, Mapping):
            return dict(raw_meta)
        raise self._malformed_response_error(
            endpoint=endpoint,
            context=context,
            expected="task field 'meta' to be a mapping",
            response=task,
        )

    def _require_mapping(self, response: Any, *, endpoint: str, context: str) -> Dict[str, Any]:
        if isinstance(response, Mapping):
            return dict(response)
        raise self._malformed_response_error(
            endpoint=endpoint,
            context=context,
            expected="mapping response",
            response=response,
        )

    def _require_id(self, response: Any, *, endpoint: str, context: str) -> str:
        mapping = self._require_mapping(response, endpoint=endpoint, context=context)
        raw_id = mapping.get("id")
        if raw_id is None or str(raw_id).strip() == "":
            raise self._malformed_response_error(
                endpoint=endpoint,
                context=context,
                expected="mapping response with non-empty 'id'",
                response=mapping,
            )
        return str(raw_id)

    def _malformed_response_error(self, *, endpoint: str, context: str, expected: str, response: Any) -> LabelBackendError:
        return LabelBackendError(
            "Malformed Label Studio API response: "
            f"endpoint={endpoint!r}, context={context!r}, expected={expected}, "
            f"response_summary={self._response_summary(response)}"
        )

    def _response_summary(self, response: Any, *, depth: int = 0) -> Any:
        if depth > 3:
            return "<omitted>"
        if isinstance(response, Mapping):
            result: Dict[str, Any] = {}
            for index, (key, value) in enumerate(response.items()):
                if index >= 10:
                    result["<truncated>"] = True
                    break
                key_str = str(key)
                if any(marker in key_str.lower() for marker in ("token", "password", "secret", "authorization")):
                    result[key_str] = "<redacted>"
                else:
                    result[key_str] = self._response_summary(value, depth=depth + 1)
            return result
        if isinstance(response, list):
            items = [self._response_summary(item, depth=depth + 1) for item in response[:10]]
            if len(response) > 10:
                items.append("<truncated>")
            return items
        if isinstance(response, str):
            return response if len(response) <= 300 else response[:300] + "...<truncated>"
        if response is None or isinstance(response, (bool, int, float)):
            return response
        return repr(response)

    def _has_next_page(self, response: Any) -> bool:
        return isinstance(response, Mapping) and bool(response.get("next"))

    def _extract_annotations(self, task: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        raw_annotations = task.get("annotations", []) or []
        if not isinstance(raw_annotations, list):
            raise self._malformed_response_error(
                endpoint="/api/tasks/<id>/",
                context="extract annotations",
                expected="field 'annotations' to be a list",
                response=task,
            )
        annotations = []
        for index, annotation in enumerate(raw_annotations):
            if not isinstance(annotation, Mapping):
                raise self._malformed_response_error(
                    endpoint="/api/tasks/<id>/",
                    context=f"extract annotations item[{index}]",
                    expected="annotation mapping",
                    response=annotation,
                )
            if annotation.get("was_cancelled") or annotation.get("skipped"):
                continue
            annotations.append(annotation)
        return annotations

    def _ready_vote_count(self, annotations: Sequence[Mapping[str, Any]], policy: AnnotationPolicy) -> int:
        parsed_annotations = [
            record
            for record in (self._parse_annotation(annotation) for annotation in annotations)
            if record is not None
        ]
        if policy.allow_single_annotator:
            return len(parsed_annotations)
        annotator_ids = set()
        for record in parsed_annotations:
            annotator_ids.add(record.annotator_id)
        return len(annotator_ids)

    def _parse_annotation(self, annotation: Mapping[str, Any]) -> Optional[AnnotationRecord]:
        values: List[Any] = []
        results = annotation.get("result", []) or []
        if not isinstance(results, list):
            raise self._malformed_response_error(
                endpoint="/api/tasks/<id>/",
                context="parse annotation result",
                expected="field 'result' to be a list",
                response=annotation,
            )
        for index, result in enumerate(results):
            if not isinstance(result, Mapping):
                raise self._malformed_response_error(
                    endpoint="/api/tasks/<id>/",
                    context=f"parse annotation result[{index}]",
                    expected="result mapping",
                    response=result,
                )
            values.extend(self._extract_result_values(result))
        if not values:
            return None

        if self._label_schema is not None and self._label_schema.multi_label:
            value: Any = values
        else:
            value = values[-1]

        completed_by = annotation.get("completed_by")
        if isinstance(completed_by, Mapping):
            annotator_id = str(completed_by.get("email") or completed_by.get("id") or "label_studio")
        else:
            annotator_id = str(completed_by or annotation.get("created_username") or "label_studio")

        return AnnotationRecord(
            annotator_id=annotator_id,
            created_at=_parse_timestamp(annotation.get("created_at") or annotation.get("updated_at")),
            value=value,
            score=self._coerce_optional_float(annotation.get("score")),
        )

    def _extract_result_values(self, result: Mapping[str, Any]) -> List[Any]:
        if result.get("from_name") not in {None, _CONTROL_NAME}:
            return []
        if result.get("to_name") not in {None, _OBJECT_NAME}:
            return []
        value = result.get("value") or {}
        if not isinstance(value, Mapping):
            return []

        collected: List[Any] = []
        for key in ("choices", "labels"):
            option_values = value.get(key)
            if isinstance(option_values, list):
                collected.extend(option_values)

        taxonomy_values = value.get("taxonomy")
        if isinstance(taxonomy_values, list):
            for item in taxonomy_values:
                if isinstance(item, list) and item:
                    collected.append(item[-1])
                elif item:
                    collected.append(item)
        return [item for item in collected if self._is_schema_label(item)]

    def _is_schema_label(self, value: Any) -> bool:
        if self._label_schema is None:
            return True
        return value in set(self._label_schema.labels)

    def _validate_prediction_choices(self, choices: Sequence[Any]) -> None:
        if self._label_schema is None:
            return
        invalid = [choice for choice in choices if choice not in set(self._label_schema.labels)]
        if invalid:
            raise LabelBackendError(
                f"Label Studio prelabel contains labels outside label_schema: {invalid!r}."
            )

    def _validate_probability_row(self, values: Sequence[Any]) -> List[float]:
        if self._label_schema is None:
            raise LabelBackendError("Label schema is not initialized.")
        expected_width = len(self._label_schema.labels)
        if len(values) != expected_width:
            raise LabelBackendError(
                "Label Studio probability prelabel width must match label_schema: "
                f"got {len(values)}, expected {expected_width}."
            )
        if len(values) < 2:
            raise LabelBackendError("Label Studio probability prelabel must contain at least two probabilities.")

        probabilities: List[float] = []
        for value in values:
            if isinstance(value, bool):
                raise LabelBackendError("Label Studio probability prelabel values must be numeric probabilities.")
            probability = self._coerce_optional_float(value)
            if probability is None:
                raise LabelBackendError(
                    "Label Studio probability prelabel values must be finite numeric probabilities."
                )
            if probability < 0.0:
                raise LabelBackendError("Label Studio probability prelabel values must be non-negative.")
            probabilities.append(probability)

        total = math.fsum(probabilities)
        if total <= 0.0:
            raise LabelBackendError("Label Studio probability prelabel row must have a positive sum.")
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise LabelBackendError(
                f"Label Studio probability prelabel row must sum to 1.0, got {total!r}."
            )
        return probabilities

    def _coerce_prediction_score(self, value: Any) -> float:
        if isinstance(value, bool):
            raise LabelBackendError("Label Studio mapping prelabel scores must be finite non-negative numbers, not bool.")
        score = self._coerce_optional_float(value)
        if score is None:
            raise LabelBackendError("Label Studio mapping prelabel scores must be finite numeric values.")
        if score < 0.0:
            raise LabelBackendError("Label Studio mapping prelabel scores must be non-negative.")
        return score

    def _coerce_optional_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(coerced):
            return None
        return coerced
