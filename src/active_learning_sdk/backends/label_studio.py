from __future__ import annotations

"""
Label Studio backend scaffold.

This file shows where Label Studio integration code should live.

For juniors implementing this:
- Implement push_round/poll_round/pull_round using Label Studio HTTP APIs.
- Keep it idempotent: pushing the same (round_id, sample_id) twice must NOT create
  duplicate tasks.
- Normalize Label Studio JSON into `AnnotationRecord` objects on pull.
"""

from typing import Any, Dict, Mapping, Optional, Sequence

from ..configs import AnnotationPolicy, LabelBackendConfig, LabelSchema
from ..exceptions import ConfigurationError, LabelBackendError
from ..types import DataSample
from .base import RoundProgress, RoundPullResult, RoundPushResult


class LabelStudioBackend:
    """
    Label Studio backend scaffold (not fully implemented).

    Required behavior:
    - ensure_ready(): create or reuse a Label Studio project and label schema
    - push_round(): create tasks and return task_ids (sample_id -> task_id)
    - poll_round(): check if annotations are ready according to AnnotationPolicy
    - pull_round(): fetch annotations and parse them into AnnotationRecord

    Attributes:
        config (LabelBackendConfig):
            Where: set in `__init__` and used by all API calls.
            What: Label Studio connection/project settings (base_url, api_token, project_id, etc).
            Why: keeps integration parameters in one place and makes behavior reproducible.
        _ready (bool):
            Where: flipped in `ensure_ready()` and checked in push/poll/pull.
            What: internal flag "backend has been initialized".
            Why: prevents accidental calls before authentication/project setup.
        _project_ref (Dict[str, Any]):
            Where: returned from `ensure_ready()` and can be saved for debugging.
            What: backend-specific metadata (project id, urls, etc).
            Why: helps troubleshooting and makes it possible to reuse existing backend resources.
        _requests (Any):
            Where: set in `__init__` when importing `requests`.
            What: the imported `requests` module (or None if import failed).
            Why: keeps the hard dependency localized to this backend implementation.
    """

    def __init__(self, config: LabelBackendConfig) -> None:
        config.validate()
        self.config = config
        self._ready = False
        self._project_ref: Dict[str, Any] = {}
        self._requests = None
        try:
            import requests  # type: ignore

            self._requests = requests
        except Exception as error:
            raise ConfigurationError(
                "LabelStudioBackend requires 'requests'. Install active-learning-sdk[labelstudio] or requests."
            ) from error

    def ensure_ready(self, label_schema: LabelSchema) -> Dict[str, Any]:
        """
        Prepare Label Studio to accept tasks for this labeling schema.

        In a real implementation this typically:
        - authenticates using api_token
        - creates or reuses a project
        - configures the labeling interface
        """
        label_schema.validate()
        self._ready = True
        self._project_ref = {"backend": "label_studio", "project_id": self.config.project_id}
        return dict(self._project_ref)

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: Optional[Dict[str, Any]] = None,
    ) -> RoundPushResult:
        """
        Create Label Studio tasks for the selected samples.

        Must be idempotent:
        - If this method is called again for the same round, it should reuse existing tasks
          (or at least return the same mapping) instead of creating duplicates.
        """
        if not self._ready:
            raise LabelBackendError("LabelStudioBackend is not ready. Call ensure_ready() first.")
        raise NotImplementedError("Implement HTTP calls to Label Studio API here.")

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        """
        Check progress for tasks in the backend.

        The engine uses this to decide when it can pull annotations.
        Typical logic:
        - count how many tasks have >= policy.min_votes annotations
        - return RoundProgress(total=..., done=..., ready_sample_ids=...)
        """
        if not self._ready:
            raise LabelBackendError("LabelStudioBackend is not ready. Call ensure_ready() first.")
        raise NotImplementedError("Implement polling logic against Label Studio API here.")

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        """
        Fetch annotations for tasks and convert them to `AnnotationRecord`.

        Output format required by the engine:
        - dict[sample_id] -> list[AnnotationRecord]
        """
        if not self._ready:
            raise LabelBackendError("LabelStudioBackend is not ready. Call ensure_ready() first.")
        raise NotImplementedError("Implement pull logic against Label Studio API here.")

    def close(self) -> None:
        self._ready = False
        self._project_ref = {}
