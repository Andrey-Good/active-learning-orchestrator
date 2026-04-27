from __future__ import annotations

"""
Core data structures and enums used across the SDK.

These types are shared by multiple subsystems (engine, state, backends, dataset, etc).
If you are adding a new feature, prefer reusing these types instead of inventing
new ad-hoc dict formats.
"""

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
import numpy as np

class SampleStatus(str, enum.Enum):
    """Lifecycle status of a sample inside the project.

    The SDK keeps the original MVP values for backward compatibility and adds the
    PRD lifecycle values used by the Phase 1 state model. The PRD flow is:

        UNLABELED -> SCORED -> PENDING -> SENT -> LABELED -> IN_TRAINING
    """

    UNLABELED = "unlabeled"
    SCORED = "scored"
    PENDING = "pending"
    SENT = "sent"
    LABELED = "labeled"
    IN_TRAINING = "in_training"
    NEEDS_REVIEW = "needs_review"
    INVALID = "invalid"

    @classmethod
    def selectable_values(cls) -> set[str]:
        """Statuses that may enter a new selection round."""
        return {cls.UNLABELED.value, cls.SCORED.value}

    @classmethod
    def training_ready_values(cls) -> set[str]:
        """Statuses that have labels and may be used for training/evaluation."""
        return {cls.LABELED.value, cls.IN_TRAINING.value}

    @classmethod
    def terminal_values(cls) -> set[str]:
        """Statuses that should not re-enter selection without manual reset."""
        return {cls.IN_TRAINING.value, cls.NEEDS_REVIEW.value, cls.INVALID.value}


PRD_SAMPLE_LIFECYCLE: tuple[SampleStatus, ...] = (
    SampleStatus.UNLABELED,
    SampleStatus.SCORED,
    SampleStatus.PENDING,
    SampleStatus.SENT,
    SampleStatus.LABELED,
    SampleStatus.IN_TRAINING,
)

PRD_LIFECYCLE_ALIASES: dict[str, SampleStatus] = {
    "new_unseen": SampleStatus.UNLABELED,
    "unlabeled": SampleStatus.UNLABELED,
    "scored": SampleStatus.SCORED,
    "pending": SampleStatus.PENDING,
    "sent": SampleStatus.SENT,
    "labeled": SampleStatus.LABELED,
    "in_training": SampleStatus.IN_TRAINING,
    "needs_review": SampleStatus.NEEDS_REVIEW,
    "invalid": SampleStatus.INVALID,
}


class RoundStatus(str, enum.Enum):
    """Lifecycle status of a round. Used for idempotent resume."""

    SELECTING = "selecting"
    SELECTED = "selected"
    PUSHED = "pushed"
    WAITING = "waiting"
    READY_TO_PULL = "ready_to_pull"
    PULLED = "pulled"
    TRAINED = "trained"
    DONE = "done"
    FAILED = "failed"


class StepKind(str, enum.Enum):
    """The next executable step that run_step() may perform."""

    SELECT = "select"
    PUSH = "push"
    WAIT = "wait"
    PULL = "pull"
    TRAIN_EVAL = "train_eval"
    UPDATE = "update"
    NOOP = "noop"


@dataclass(frozen=True)
class DataSample:
    """
    One dataset item inside the SDK.

    `sample_id` must be stable across runs, because it is used as the key in state.json.

    For the text MVP we typically store:
    - data={"text": "..."}

    `meta` can store any extra info you want (source, tags, etc).

    Attributes:
        sample_id (str):
            Stable identifier of the sample.
            Where: used as the key in `state.json` (statuses, labels, caches).
            What: a string, unique within the dataset.
            Why: stability is required for resume and idempotency.
        data (Dict[str, Any]):
            Main payload of the sample.
            Where: passed to backends, model adapters, and strategies.
            What: for MVP should contain "text".
            Why: keeps the SDK modality-agnostic (future: images/audio/etc).
        meta (Dict[str, Any]):
            Optional metadata.
            Where: can be exported, shown in labeling UI, used for analysis.
            What: any JSON-serializable info.
            Why: avoids losing useful context while keeping `data` minimal.
        group_id (Optional[str]):
            Optional grouping key.
            Where: can be used by future strategies (diversity, constraints).
            What: string or None.
            Why: helps enforce "do not pick too many from same group" patterns.
    """

    sample_id: str
    data: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)
    group_id: Optional[str] = None

class VectorStore(Protocol):
    def add(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Must be able to save vectors."""
        ...
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Must be able to search for similar texts."""
        ...

class InferenceRunner(Protocol):
    def run_inference(self, pool_texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Must be able to accept texts and return probabilities (how confident it is in each class)."""
        ...

@dataclass(frozen=True)
class AnnotationRecord:
    """
    One annotation for a sample.

    A backend returns `AnnotationRecord` objects after parsing its own payload.
    The engine later aggregates them into a single `ResolvedLabel`.

    Attributes:
        annotator_id (str):
            Who produced the annotation.
            Where: used for debugging and future multi-annotator logic.
            What: user id, email, or "llm".
            Why: lets you track sources and measure agreement.
        created_at (float):
            Timestamp when the annotation was created.
            Where: used to sort annotations deterministically.
            What: seconds since epoch.
            Why: makes "latest" mode well-defined.
        value (Any):
            The raw label value.
            Where: interpreted by your task schema (single label, multi-label list, etc).
            What: usually str or list[str] for classification.
            Why: keeps the SDK flexible across tasks.
        score (Optional[float]):
            Optional confidence/agreement info from backend.
            Where: can be used for review workflows.
            What: number or None.
            Why: supports richer policies in the future.
    """

    annotator_id: str
    created_at: float
    value: Any
    score: Optional[float] = None


@dataclass(frozen=True)
class ResolvedLabel:
    """
    Final label produced after applying `AnnotationPolicy`.

    This is what gets written into:
    - `ProjectState.sample_labels`
    - `ProjectState.sample_status`

    Attributes:
        sample_id (str):
            Which dataset sample this label belongs to.
            Where: used as key in state.
            What: same id as in `DataSample.sample_id`.
            Why: ties resolved results back to dataset.
        status (SampleStatus):
            Whether the sample is labeled, needs review, etc.
            Where: engine updates `ProjectState.sample_status`.
            What: LABELED / NEEDS_REVIEW / UNLABELED / INVALID.
            Why: controls which pool the sample belongs to next.
        label (Optional[Any]):
            Final label used for training.
            Where: saved in `ProjectState.sample_labels`.
            What: often a string label name.
            Why: model training uses this as target.
        agreement (Optional[float]):
            Agreement ratio for majority/consensus decisions.
            Where: useful for audit/debug and review logic.
            What: 0..1 or None.
            Why: tells you how confident the aggregation was.
        details (Dict[str, Any]):
            Extra info about the decision.
            Where: debugging, analytics, UI.
            What: counts, reasons, etc.
            Why: makes it easier to understand why a sample is NEEDS_REVIEW.
    """

    sample_id: str
    status: SampleStatus
    label: Optional[Any] = None
    agreement: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricRecord:
    """
    Metrics snapshot (usually produced after TRAIN_EVAL step).

    `metrics` is a simple dict like {"accuracy": 0.91}.

    Attributes:
        step (str):
            Which stage produced the metrics.
            Where: engine uses "eval" in TRAIN_EVAL step.
            What: string tag.
            Why: allows storing multiple metric types in the future.
        created_at (float):
            Timestamp when metrics were recorded.
            Where: timeline plots/reports.
            What: seconds since epoch.
            Why: helps debug training history.
        metrics (Dict[str, float]):
            Metric name -> value.
            Where: used for reward computation and reporting.
            What: e.g. {"accuracy": 0.9, "f1": 0.88}.
            Why: active learning needs feedback signals to measure progress.
    """

    step: str
    created_at: float
    metrics: Dict[str, float]

@dataclass
class StoppingResult:
    should_stop: bool
    confidence_level: float

class StoppingCriterion(Protocol):
    def check_stop(self, current_metrics: Dict, history: List[Dict]) -> StoppingResult:
        """Any recovery algorithm must be able to make a verdict."""
        ...

class Calibrator(Protocol):
    def calibrate(self, raw_logits: np.ndarray) -> np.ndarray:
        """Makes the neural network's probabilities fair."""
        ...