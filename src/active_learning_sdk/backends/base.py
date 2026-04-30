"""
Labeling backend interfaces and small helpers.

The engine talks to "a backend" through the `LabelBackend` protocol.
This keeps the engine independent from any specific tool (Label Studio, custom UI, LLM).

For juniors:
- If you want the SDK to work with a real labeling tool, you implement a backend.
- The engine already knows when to call push/poll/pull; you only implement the I/O.
"""

from __future__ import annotations


import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from ..configs import AnnotationPolicy, LabelBackendConfig, LabelSchema
from ..exceptions import ConfigurationError, LabelBackendError
from ..types import AnnotationRecord, DataSample


@dataclass(frozen=True)
class RoundPushResult:
    """
    Return value of `LabelBackend.push_round()`.

    Attributes:
        task_ids (Dict[str, str]):
            Where: persisted into `RoundState.task_ids` by the engine.
            What: mapping sample_id -> backend task id.
            Why: this is the idempotency anchor (prevents duplicate task creation on resume).
        backend_round_ref (Dict[str, Any]):
            Where: optional debug info saved/returned by the backend.
            What: backend-specific metadata (project id, urls, etc).
            Why: helps debugging integration issues.
    """
    task_ids: Dict[str, str]
    backend_round_ref: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundProgress:
    """
    Return value of `LabelBackend.poll_round()`.

    Attributes:
        total (int):
            Where: used by engine to decide if everything is ready.
            What: how many tasks are being tracked.
            Why: provides a stable denominator for progress.
        done (int):
            Where: used by engine to transition READY_TO_PULL.
            What: how many tasks are ready according to policy.
            Why: indicates when it is safe to pull.
        ready_sample_ids (List[str]):
            Where: optional; can be used for partial pulling in the future.
            What: sample ids that are ready now.
            Why: useful if you want to support streaming/partial progress.
        details (Dict[str, Any]):
            Where: debug/monitoring.
            What: backend-specific payload (counts, per-task states, etc).
            Why: helps when diagnosing why WAIT is slow.
    """
    total: int
    done: int
    ready_sample_ids: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundPullResult:
    """
    Return value of `LabelBackend.pull_round()`.

    Attributes:
        annotations (Dict[str, List[AnnotationRecord]]):
            Where: consumed by `AnnotationAggregator.resolve(...)`.
            What: mapping sample_id -> list of AnnotationRecord.
            Why: backends can produce multiple annotations per sample.
        backend_payload (Dict[str, Any]):
            Where: debug/monitoring.
            What: raw backend response fragments (optional).
            Why: helps implementers validate parsing logic.
    """
    annotations: Dict[str, List[AnnotationRecord]]
    backend_payload: Dict[str, Any] = field(default_factory=dict)


class LabelBackend(Protocol):
    """
    Contract for labeling backends.

    The engine calls these methods in this order:
    1. ensure_ready() once per process
    2. push_round() for a new round
    3. poll_round() repeatedly until enough labels are ready
    4. pull_round() to fetch annotations for training

    Idempotency rule (very important):
    - `push_round()` should not create duplicates if the engine retries after a crash.
      Use `round_id` + `sample_id` as a stable external id when possible.

    Attributes:
        (backend-specific):
            Where: each backend implementation stores its own connection/session state.
            What: for example HTTP session, project id, auth token, etc.
            Why: the engine treats the backend as a black box and only calls the methods.
    """

    def ensure_ready(self, label_schema: LabelSchema) -> Dict[str, Any]:
        ...

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: Optional[Dict[str, Any]] = None,
    ) -> RoundPushResult:
        ...

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        ...

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        ...

    def close(self) -> None:
        ...


class LLMLabelBackend:
    """
    Placeholder backend for programmatic/LLM labeling.

    This backend is synchronous:
    - `poll_round()` always returns "done"
    - `pull_round()` calls the provided `label_fn`

    Attributes:
        _label_fn (Callable[[DataSample], AnnotationRecord]):
            Where: called inside `pull_round()` to create an annotation.
            What: user-provided function that returns one AnnotationRecord for a sample.
            Why: lets you plug in an LLM or rules-based labeler without Label Studio.
        _ready (bool):
            Where: set in `ensure_ready()` and checked in `pull_round()`.
            What: "initialization complete" flag.
            Why: mirrors the contract of real backends (helps catch misuse early).
    """

    def __init__(self, label_fn: Callable[[DataSample], AnnotationRecord]) -> None:
        self._label_fn = label_fn
        self._ready = False
        self._samples_by_round: Dict[str, Dict[str, DataSample]] = {}

    def ensure_ready(self, label_schema: LabelSchema) -> Dict[str, Any]:
        label_schema.validate()
        self._ready = True
        return {"backend": "llm"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: Optional[Dict[str, Any]] = None,
    ) -> RoundPushResult:
        self._samples_by_round[round_id] = {sample.sample_id: sample for sample in samples}
        task_ids = {sample.sample_id: f"llm:{round_id}:{sample.sample_id}" for sample in samples}
        return RoundPushResult(task_ids=task_ids, backend_round_ref={"round_id": round_id})

    def restore_round_samples(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        task_ids: Optional[Mapping[str, str]] = None,
    ) -> None:
        restored = {sample.sample_id: sample for sample in samples}
        if task_ids is not None:
            for sample_id in restored:
                self._validate_task_binding(round_id, sample_id, task_ids.get(sample_id, ""))
        self._samples_by_round[round_id] = restored

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        for sample_id, task_id in task_ids.items():
            self._validate_task_binding(round_id, sample_id, task_id)
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids.keys()))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        if not self._ready:
            raise LabelBackendError("LLMLabelBackend is not ready. Call ensure_ready() first.")
        annotations: Dict[str, List[AnnotationRecord]] = {}
        now = time.time()
        pushed_samples = self._samples_by_round.get(round_id, {})
        for sample_id, task_id in task_ids.items():
            self._validate_task_binding(round_id, sample_id, task_id)
            sample = pushed_samples.get(sample_id)
            if sample is None:
                raise LabelBackendError(
                    f"LLMLabelBackend has no pushed sample payload for round_id={round_id!r}, sample_id={sample_id!r}."
                )
            annotation = self._label_fn(sample)
            if annotation.created_at <= 0:
                annotation = AnnotationRecord(
                    annotator_id=annotation.annotator_id,
                    created_at=now,
                    value=annotation.value,
                    score=annotation.score,
                )
            annotations[sample_id] = [annotation]
        return RoundPullResult(annotations=annotations)

    def _validate_task_binding(self, round_id: str, sample_id: str, task_id: str) -> None:
        expected = f"llm:{round_id}:{sample_id}"
        if str(task_id) != expected:
            raise LabelBackendError(
                f"LLM task id {task_id!r} does not belong to sample_id={sample_id!r} in round_id={round_id!r}."
            )

    def close(self) -> None:
        self._ready = False
        self._samples_by_round.clear()


def build_label_backend(config: LabelBackendConfig) -> LabelBackend:
    """
    Factory that builds a backend from config.

    Supported built-ins:
    - Label Studio
    - simulator (deterministic test backend)
    """
    if config.backend == "label_studio":
        from .label_studio import LabelStudioBackend

        return LabelStudioBackend(config)
    if config.backend == "simulator":
        from .simulator import SimulatorLabelBackend

        return SimulatorLabelBackend()
    if config.backend == "llm":
        raise ConfigurationError("LLM backend requires a label function; instantiate LLMLabelBackend directly.")
    if config.backend == "custom":
        raise ConfigurationError(
            "Custom label backend configs require injecting/providing a label_backend instance; "
            "build_label_backend() only constructs built-in backends."
        )
    raise ConfigurationError(f"Unsupported backend: {config.backend!r}")
