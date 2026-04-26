from __future__ import annotations

"""
Persistent project state (state.json).

This module defines the dataclasses that represent the on-disk state and the code that
loads/saves it.

For juniors:
- The state is the "memory" of an active learning project.
- The engine writes it after each step so the project can resume after crashes.
- Most idempotency guarantees come from fields stored here (for example task_ids).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from ..exceptions import StateCorruptedError
from ..types import MetricRecord, RoundStatus
from ..utils import atomic_write_text, dataclass_to_dict


@dataclass
class DatasetRef:
    """
    Information about which dataset was used and how to verify it.

    The important part is `fingerprint`: the engine checks it on resume to make sure
    you did not accidentally swap the dataset for the same `workdir`.

    Attributes:
        source_type (str):
            What kind of dataset source this was.
            Where: saved into state during `configure()`.
            What: "provider", "dataframe", or "path" (engine uses these values).
            Why: helps debugging and reporting.
        source_path (Optional[str]):
            Original dataset path if the dataset was provided as a path.
            Where: persisted in state for humans.
            What: filesystem path string or None.
            Why: makes it easier to reproduce runs.
        schema (Dict[str, str]):
            Stable schema description from provider.
            Where: included in fingerprint inputs.
            What: e.g. {"sample_id": "str", "text": "str"}.
            Why: schema changes should invalidate resume.
        fingerprint (str):
            Deterministic fingerprint of the dataset.
            Where: compared during `configure()` and `attach_runtime()`.
            What: hex digest string.
            Why: prevents mixing different datasets in the same project directory.
        fingerprint_config (Dict[str, Any]):
            The exact fingerprint config used to compute `fingerprint`.
            Where: stored so attach_runtime can recompute the same way.
            What: dict produced by `dataclasses.asdict(FingerprintConfig)`.
            Why: fingerprint must be reproducible across processes.
    """

    source_type: str
    source_path: Optional[str]
    schema: Dict[str, str]
    fingerprint: str
    fingerprint_config: Dict[str, Any]


@dataclass
class RoundState:
    """
    Persisted state for one active learning round.

    Key fields (very important for resume):
    - selected_sample_ids: what was chosen in SELECT
    - task_ids: mapping sample_id -> backend task id (idempotency anchor)
    - status: where we are in the state machine

    Attributes:
        round_id (str):
            Unique round identifier.
            Where: used when creating tasks and storing history.
            What: string like "r0001_ab12cd34".
            Why: makes resume/debugging easier and supports idempotency.
        status (RoundStatus):
            Current stage of the round.
            Where: used by engine to choose the next step.
            What: SELECTING/SELECTED/PUSHED/WAITING/READY_TO_PULL/PULLED/TRAINED/DONE/FAILED.
            Why: drives the state machine.
        created_at (float):
            When the round was created.
            Where: status views/reports.
            What: epoch timestamp.
            Why: helps monitoring and debugging.
        updated_at (float):
            Last time this round was modified.
            Where: updated after each step.
            What: epoch timestamp.
            Why: helps detect stuck rounds.
        selected_sample_ids (List[str]):
            Sample IDs chosen in SELECT step.
            Where: used in PUSH step to create tasks.
            What: list of dataset sample_id values.
            Why: selection must be persisted for deterministic resume.
        task_ids (Dict[str, str]):
            sample_id -> backend task id mapping.
            Where: set in PUSH step, used in WAIT/PULL steps.
            What: dict mapping.
            Why: idempotency anchor to avoid duplicate tasks on resume.
        resolved (Dict[str, Any]):
            sample_id -> final label value (for labeled items).
            Where: set in PULL step.
            What: dict mapping.
            Why: round-level audit of what labels were produced.
        metrics_before (Dict[str, float]):
            Metrics snapshot before training for this round.
            Where: set in TRAIN_EVAL step.
            What: dict.
            Why: used for reward computation.
        metrics_after (Dict[str, float]):
            Metrics snapshot after training/eval for this round.
            Where: set in TRAIN_EVAL step.
            What: dict.
            Why: reward computation and reporting.
        reward (Optional[float]):
            Computed reward signal for bandit schedulers.
            Where: set in UPDATE step.
            What: float or None.
            Why: future scheduling algorithms need a numeric signal.
        scheduler_snapshot (Dict[str, Any]):
            Record of which strategy/mode was used to select this round.
            Where: set in SELECT step.
            What: dict like {"mode": "single", "strategy": "entropy"}.
            Why: reproducibility and debugging.
        error (Optional[str]):
            Error string if round failed.
            Where: not heavily used in scaffold yet.
            What: message string or None.
            Why: helps diagnosing failures after the fact.
    """

    round_id: str
    status: RoundStatus
    created_at: float
    updated_at: float
    selected_sample_ids: List[str] = field(default_factory=list)
    task_ids: Dict[str, str] = field(default_factory=dict)
    resolved: Dict[str, Any] = field(default_factory=dict)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    reward: Optional[float] = None
    scheduler_snapshot: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ProjectState:
    """
    Persisted project state (JSON-serializable).

    This is stored in `workdir/state.json`.
    If you are debugging, opening that file can be very helpful.

    Attributes:
        state_version (int):
            Schema version of this state file.
            Where: checked on engine startup.
            What: integer constant.
            Why: supports future migrations.
        project_name (str):
            Human-readable name of the project.
            Where: validated on open to avoid mixing projects in one folder.
            What: string.
            Why: sanity check and reporting.
        created_at (float):
            When the project was created.
            Where: status/report.
            What: epoch timestamp.
            Why: timeline/debug.
        updated_at (float):
            Last time any state was saved.
            Where: updated on every checkpoint.
            What: epoch timestamp.
            Why: monitoring/staleness detection.
        dataset_ref (Optional[DatasetRef]):
            Dataset identity and fingerprint info.
            Where: set in `configure`.
            What: DatasetRef or None (before configure).
            Why: resume safety.
        label_schema (Optional[Dict[str, Any]]):
            Persisted label schema as a plain dict.
            Where: set in `configure`, reconstructed in `attach_runtime`.
            What: dict from `dataclasses.asdict(LabelSchema)`.
            Why: allows restarting without retyping configs.
        annotation_policy (Dict[str, Any]):
            Persisted policy config as dict.
            Where: used to rebuild AnnotationPolicy on resume.
            Why: determinism and reproducibility.
        scheduler_config (Dict[str, Any]):
            Persisted scheduler config as dict.
            Where: used to rebuild SchedulerConfig on resume.
        label_backend_config (Dict[str, Any]):
            Persisted backend config as dict.
            Where: used to rebuild LabelBackendConfig on resume.
        cache_config (Dict[str, Any]):
            Persisted caching settings as dict.
            Where: used to rebuild CacheConfig on resume.
        split_config (Dict[str, Any]):
            Persisted split settings as dict.
            Where: used to document how splits were made.
        prelabel_config (Dict[str, Any]):
            Persisted prelabel settings as dict.
        splits (Dict[str, List[str]]):
            Explicit split IDs.
            Where: created in configure via engine `_resolve_splits`.
            Why: ensures deterministic train/val selection.
        sample_status (Dict[str, str]):
            sample_id -> SampleStatus.value.
            Where: core tracking for which samples are unlabeled/labeled.
            Why: selection pool is derived from this map.
        sample_labels (Dict[str, Any]):
            sample_id -> final label value.
            Where: training labels are read from this map.
        rounds (List[RoundState]):
            History of rounds.
            Where: used for resume and reporting.
        metrics_history (List[MetricRecord]):
            Metrics timeline.
            Where: reporting and reward computation.
        scheduler_state (Dict[str, Any]):
            Internal scheduler state (for bandits).
            Where: updated in UPDATE step.
        caches_index (Dict[str, Any]):
            Reserved bookkeeping for caches.
            Where: not required in scaffold, but kept for future expansion.
    """

    state_version: int
    project_name: str
    created_at: float
    updated_at: float
    dataset_ref: Optional[DatasetRef] = None
    label_schema: Optional[Dict[str, Any]] = None
    annotation_policy: Dict[str, Any] = field(default_factory=dict)
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    label_backend_config: Dict[str, Any] = field(default_factory=dict)
    cache_config: Dict[str, Any] = field(default_factory=dict)
    split_config: Dict[str, Any] = field(default_factory=dict)
    prelabel_config: Dict[str, Any] = field(default_factory=dict)
    splits: Dict[str, List[str]] = field(default_factory=dict)
    sample_status: Dict[str, str] = field(default_factory=dict)
    sample_labels: Dict[str, Any] = field(default_factory=dict)
    rounds: List[RoundState] = field(default_factory=list)
    metrics_history: List[MetricRecord] = field(default_factory=list)
    scheduler_state: Dict[str, Any] = field(default_factory=dict)
    caches_index: Dict[str, Any] = field(default_factory=dict)


def _from_dict_roundstate(payload: Dict[str, Any]) -> RoundState:
    return RoundState(
        round_id=payload["round_id"],
        status=RoundStatus(payload["status"]),
        created_at=float(payload["created_at"]),
        updated_at=float(payload["updated_at"]),
        selected_sample_ids=list(payload.get("selected_sample_ids", [])),
        task_ids=dict(payload.get("task_ids", {})),
        resolved=dict(payload.get("resolved", {})),
        metrics_before=dict(payload.get("metrics_before", {})),
        metrics_after=dict(payload.get("metrics_after", {})),
        reward=payload.get("reward"),
        scheduler_snapshot=dict(payload.get("scheduler_snapshot", {})),
        error=payload.get("error"),
    )


def _from_dict_dataset_ref(payload: Dict[str, Any]) -> DatasetRef:
    return DatasetRef(
        source_type=payload["source_type"],
        source_path=payload.get("source_path"),
        schema=dict(payload.get("schema", {})),
        fingerprint=payload["fingerprint"],
        fingerprint_config=dict(payload.get("fingerprint_config", {})),
    )


def _from_dict_metric_record(payload: Dict[str, Any]) -> MetricRecord:
    return MetricRecord(
        step=payload["step"],
        created_at=float(payload["created_at"]),
        metrics=dict(payload["metrics"]),
    )


def state_to_json_dict(state: ProjectState) -> Dict[str, Any]:
    """Serialize ProjectState into plain dicts/lists/enums for JSON dumping."""
    return dataclass_to_dict(state)


def state_from_json_dict(payload: Dict[str, Any]) -> ProjectState:
    """Parse dict loaded from JSON into ProjectState dataclasses."""
    try:
        dataset_ref = _from_dict_dataset_ref(payload["dataset_ref"]) if payload.get("dataset_ref") else None
        rounds = [_from_dict_roundstate(item) for item in payload.get("rounds", [])]
        metrics_history = [_from_dict_metric_record(item) for item in payload.get("metrics_history", [])]
        return ProjectState(
            state_version=int(payload["state_version"]),
            project_name=str(payload["project_name"]),
            created_at=float(payload["created_at"]),
            updated_at=float(payload["updated_at"]),
            dataset_ref=dataset_ref,
            label_schema=payload.get("label_schema"),
            annotation_policy=dict(payload.get("annotation_policy", {})),
            scheduler_config=dict(payload.get("scheduler_config", {})),
            label_backend_config=dict(payload.get("label_backend_config", {})),
            cache_config=dict(payload.get("cache_config", {})),
            split_config=dict(payload.get("split_config", {})),
            prelabel_config=dict(payload.get("prelabel_config", {})),
            splits=dict(payload.get("splits", {})),
            sample_status=dict(payload.get("sample_status", {})),
            sample_labels=dict(payload.get("sample_labels", {})),
            rounds=rounds,
            metrics_history=metrics_history,
            scheduler_state=dict(payload.get("scheduler_state", {})),
            caches_index=dict(payload.get("caches_index", {})),
        )
    except Exception as error:
        raise StateCorruptedError(f"Failed to parse project state: {error}") from error


class StateStore(Protocol):
    """
    Storage interface for project state.

    If you want to store state in something else (SQLite, database), implement this.

    Attributes:
        (implementation-specific):
            Where: concrete stores keep their own connection/paths.
            What: for example a `state_path` (JSON file) or a DB handle (SQLite).
            Why: the engine uses the store via methods only, so it can support new backends later.
    """

    def load(self) -> ProjectState:
        ...

    def save_atomic(self, state: ProjectState) -> None:
        ...


class JsonFileStateStore:
    """
    JSON-file based state store with atomic writes.

    This is the simplest possible implementation:
    - `load()` reads the whole file into memory
    - `save_atomic()` rewrites the whole file safely

    Attributes:
        state_path (Path):
            Where: read by `load()` and written by `save_atomic()`.
            What: filesystem path to `state.json` (or another chosen file).
            Why: all project resume/idempotency depends on writing and reading this file.
    """

    def __init__(self, state_path: Union[str, Path]) -> None:
        self.state_path = Path(state_path)

    def load(self) -> ProjectState:
        """Load and parse `state.json` from disk."""
        if not self.state_path.exists():
            raise StateCorruptedError(f"State file does not exist: {self.state_path}")
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return state_from_json_dict(payload)
        except json.JSONDecodeError as error:
            raise StateCorruptedError(f"Invalid JSON in state file: {error}") from error
        except Exception as error:
            raise StateCorruptedError(f"Failed to load state file: {error}") from error

    def save_atomic(self, state: ProjectState) -> None:
        """Write the state to disk using atomic replacement."""
        serialized = json.dumps(state_to_json_dict(state), ensure_ascii=False, indent=2)
        atomic_write_text(self.state_path, serialized)
