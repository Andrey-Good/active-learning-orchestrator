"""
Deterministic in-memory backend for backend contract tests.

The simulator mirrors the real backend lifecycle:
- ensure_ready() validates schema
- push_round() creates stable task ids
- poll_round() reports readiness from submitted annotations
- pull_round() returns parsed AnnotationRecord objects
"""

from __future__ import annotations


import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..configs import AnnotationPolicy, LabelSchema
from ..exceptions import LabelBackendError
from ..types import AnnotationRecord, DataSample
from .base import RoundProgress, RoundPullResult, RoundPushResult


@dataclass
class _SimulatorTask:
    sample: DataSample
    task_id: str
    round_id: str
    annotations: List[AnnotationRecord] = field(default_factory=list)


class SimulatorLabelBackend:
    """
    In-memory backend used for deterministic tests and local simulations.

    Tests can submit annotations directly with `submit_annotation()` to emulate
    human work without relying on a live Label Studio instance.
    """

    def __init__(
        self,
        *,
        label_by_sample_id: Optional[Mapping[str, Any]] = None,
        label_fn: Optional[Callable[[DataSample], Any]] = None,
        oracle_on: str = "push",
    ) -> None:
        if oracle_on not in {"push", "pull"}:
            raise ValueError("oracle_on must be either 'push' or 'pull'.")
        if label_by_sample_id is not None and label_fn is not None:
            raise ValueError("Configure only one simulator oracle: label_by_sample_id or label_fn.")
        self._ready = False
        self._label_schema: Optional[LabelSchema] = None
        self._tasks_by_round: Dict[str, Dict[str, _SimulatorTask]] = {}
        self._tasks_by_id: Dict[str, _SimulatorTask] = {}
        self._label_by_sample_id = dict(label_by_sample_id or {})
        self._label_fn = label_fn
        self._oracle_on = oracle_on

    def ensure_ready(self, label_schema: LabelSchema) -> Dict[str, Any]:
        label_schema.validate()
        self._label_schema = label_schema
        self._ready = True
        return {"backend": "simulator"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: Optional[Dict[str, Any]] = None,
    ) -> RoundPushResult:
        del prelabels
        self._require_ready()

        round_tasks = self._tasks_by_round.setdefault(round_id, {})
        sample_ids = [str(sample.sample_id) for sample in samples]
        duplicate_sample_ids = [
            sample_id
            for sample_id, count in self._counts(sample_ids).items()
            if count > 1
        ]
        if duplicate_sample_ids:
            raise LabelBackendError(f"Duplicate simulator sample ids in push payload: {duplicate_sample_ids}")
        if round_tasks and set(round_tasks) != set(sample_ids):
            raise LabelBackendError(
                f"Simulator round_id={round_id!r} already exists with sample ids {sorted(round_tasks)}; "
                f"cannot push different sample ids {sorted(sample_ids)} for the same round."
            )
        task_ids: Dict[str, str] = {}
        for sample in samples:
            existing = round_tasks.get(sample.sample_id)
            if existing is None:
                task_id = f"sim:{round_id}:{sample.sample_id}"
                existing = _SimulatorTask(sample=sample, task_id=task_id, round_id=round_id)
                round_tasks[sample.sample_id] = existing
                self._tasks_by_id[task_id] = existing
            if self._oracle_on == "push":
                self._maybe_auto_annotate(existing)
            task_ids[sample.sample_id] = existing.task_id

        return RoundPushResult(task_ids=task_ids, backend_round_ref={"backend": "simulator", "round_id": round_id})

    def restore_round_samples(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        task_ids: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._require_ready()
        restored_task_ids = self._validate_restored_task_ids(round_id, samples, task_ids)
        round_tasks = self._tasks_by_round.setdefault(round_id, {})
        for sample in samples:
            task_id = restored_task_ids[sample.sample_id]
            existing = round_tasks.get(sample.sample_id)
            if existing is None:
                existing = _SimulatorTask(sample=sample, task_id=task_id, round_id=round_id)
                round_tasks[sample.sample_id] = existing
                self._tasks_by_id[task_id] = existing
            self._validate_task_binding(round_id, sample.sample_id, task_id, task=existing)
            if self._oracle_on == "push":
                self._maybe_auto_annotate(existing)

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        self._require_ready()
        policy.validate()
        ready_sample_ids: List[str] = []
        details: Dict[str, Any] = {}
        for sample_id, task_id in task_ids.items():
            task = self._get_task_for_mapping(round_id, sample_id, task_id)
            annotation_count = len(task.annotations)
            ready_vote_count = self._ready_vote_count(task.annotations, policy)
            if ready_vote_count >= policy.min_votes:
                ready_sample_ids.append(sample_id)
            details[sample_id] = {
                "task_id": task_id,
                "annotations": annotation_count,
                "eligible_votes": ready_vote_count,
                "round_id": round_id,
            }

        return RoundProgress(
            total=len(task_ids),
            done=len(ready_sample_ids),
            ready_sample_ids=ready_sample_ids,
            details={"tasks": details},
        )

    def _ready_vote_count(self, annotations: Sequence[AnnotationRecord], policy: AnnotationPolicy) -> int:
        if policy.allow_single_annotator:
            return len(annotations)
        return len({str(annotation.annotator_id) for annotation in annotations})

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        self._require_ready()
        annotations: Dict[str, List[AnnotationRecord]] = {}
        for sample_id, task_id in task_ids.items():
            task = self._get_task_for_mapping(round_id, sample_id, task_id)
            if self._oracle_on == "pull":
                self._maybe_auto_annotate(task)
            annotations[sample_id] = sorted(task.annotations, key=lambda item: item.created_at)

        return RoundPullResult(
            annotations=annotations,
            backend_payload={"backend": "simulator", "round_id": round_id, "task_count": len(task_ids)},
        )

    def submit_annotation(
        self,
        *,
        task_id: Optional[str] = None,
        round_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        value: Any,
        annotator_id: str = "simulator",
        created_at: Optional[float] = None,
        score: Optional[float] = None,
    ) -> AnnotationRecord:
        """
        Insert a synthetic annotation for a previously pushed task.
        """
        self._require_ready()
        task: _SimulatorTask
        if task_id is not None:
            task = self._get_task(task_id)
        elif round_id is not None and sample_id is not None:
            round_task = self._tasks_by_round.get(round_id, {}).get(sample_id)
            if round_task is None:
                raise LabelBackendError(f"No simulator task found for round_id={round_id!r}, sample_id={sample_id!r}.")
            task = round_task
        else:
            raise LabelBackendError("submit_annotation requires task_id or both round_id and sample_id.")

        record = AnnotationRecord(
            annotator_id=annotator_id,
            created_at=float(created_at if created_at is not None else time.time()),
            value=value,
            score=self._safe_score(score),
        )
        task.annotations.append(record)
        return record

    def close(self) -> None:
        self._ready = False
        self._label_schema = None

    def _get_task(self, task_id: str) -> _SimulatorTask:
        task = self._tasks_by_id.get(task_id)
        if task is None:
            raise LabelBackendError(f"Unknown simulator task id: {task_id!r}")
        return task

    def _get_task_for_mapping(self, round_id: str, sample_id: str, task_id: str) -> _SimulatorTask:
        task = self._get_task(str(task_id))
        self._validate_task_binding(round_id, sample_id, str(task_id), task=task)
        return task

    def _validate_task_binding(
        self,
        round_id: str,
        sample_id: str,
        task_id: str,
        *,
        task: Optional[_SimulatorTask] = None,
    ) -> None:
        task = task or self._get_task(str(task_id))
        if task.round_id != round_id:
            raise LabelBackendError(
                f"Simulator task id {task_id!r} belongs to round_id={task.round_id!r}, not {round_id!r}."
            )
        if task.sample.sample_id != sample_id:
            raise LabelBackendError(
                f"Simulator task id {task_id!r} does not belong to sample_id={sample_id!r}."
            )
        expected = self._expected_task_id(round_id, sample_id)
        if str(task_id) != expected:
            raise LabelBackendError(
                f"Simulator task id {task_id!r} does not match expected deterministic binding {expected!r}."
            )
        if task.task_id != expected:
            raise LabelBackendError(
                f"Simulator task {task.task_id!r} is not bound to expected deterministic id {expected!r}."
            )

    def _validate_restored_task_ids(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        task_ids: Optional[Mapping[str, str]],
    ) -> Dict[str, str]:
        sample_ids = [sample.sample_id for sample in samples]
        duplicate_sample_ids = [
            sample_id
            for sample_id, count in self._counts(sample_ids).items()
            if count > 1
        ]
        if duplicate_sample_ids:
            raise LabelBackendError(f"Duplicate simulator sample ids in restore payload: {duplicate_sample_ids}")

        if task_ids is None:
            return {sample_id: self._expected_task_id(round_id, sample_id) for sample_id in sample_ids}

        normalized_task_ids: Dict[str, str] = {}
        for raw_sample_id, raw_task_id in task_ids.items():
            sample_id = str(raw_sample_id)
            if sample_id in normalized_task_ids:
                raise LabelBackendError(f"Duplicate simulator sample id in persisted task_ids: {sample_id!r}")
            normalized_task_ids[sample_id] = str(raw_task_id)

        unexpected_sample_ids = sorted(set(normalized_task_ids) - set(sample_ids))
        if unexpected_sample_ids:
            raise LabelBackendError(
                f"Unexpected simulator task ids in persisted state for samples outside this round: "
                f"{unexpected_sample_ids}"
            )

        duplicate_task_ids = [
            task_id
            for task_id, count in self._counts(normalized_task_ids.values()).items()
            if count > 1
        ]
        if duplicate_task_ids:
            raise LabelBackendError(f"Duplicate simulator task ids in persisted state: {duplicate_task_ids}")

        restored_task_ids: Dict[str, str] = {}
        for sample_id in sample_ids:
            if sample_id not in normalized_task_ids:
                raise LabelBackendError(
                    f"Missing simulator task id for round_id={round_id!r}, sample_id={sample_id!r}."
                )
            task_id = normalized_task_ids[sample_id]
            expected = self._expected_task_id(round_id, sample_id)
            if task_id != expected:
                raise LabelBackendError(
                    f"Simulator task id {task_id!r} does not match expected deterministic binding {expected!r} "
                    f"for round_id={round_id!r}, sample_id={sample_id!r}."
                )
            restored_task_ids[sample_id] = task_id
        return restored_task_ids

    def _expected_task_id(self, round_id: str, sample_id: str) -> str:
        return f"sim:{round_id}:{sample_id}"

    def _counts(self, values: Iterable[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return counts

    def _require_ready(self) -> None:
        if not self._ready:
            raise LabelBackendError("SimulatorLabelBackend is not ready. Call ensure_ready() first.")

    def _maybe_auto_annotate(self, task: _SimulatorTask) -> None:
        if task.annotations:
            return
        oracle_value = self._oracle_value(task.sample)
        if oracle_value is None:
            return
        if isinstance(oracle_value, AnnotationRecord):
            task.annotations.append(oracle_value)
            return
        task.annotations.append(
            AnnotationRecord(
                annotator_id="simulator_oracle",
                created_at=0.0,
                value=oracle_value,
                score=None,
            )
        )

    def _oracle_value(self, sample: DataSample) -> Any:
        if self._label_fn is not None:
            return self._label_fn(sample)
        return self._label_by_sample_id.get(sample.sample_id)

    def _safe_score(self, score: Optional[float]) -> Optional[float]:
        if score is None:
            return None
        try:
            value = float(score)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
