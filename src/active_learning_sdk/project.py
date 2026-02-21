from __future__ import annotations

"""
User-facing facade for the Active Learning SDK.

If you are new to this repo, start here.

`ActiveLearningProject` is intentionally thin: it mostly forwards calls to
`ActiveLearningEngine` (see `src/active_learning_sdk/engine.py`).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .adapters.base import TextClassificationAdapter
from .backends.base import LabelBackend
from .configs import (
    AnnotationPolicy,
    CacheConfig,
    FingerprintConfig,
    LabelBackendConfig,
    LabelSchema,
    PrelabelConfig,
    SchedulerConfig,
    SplitConfig,
    StopCriteria,
)
from .dataset.provider import DatasetProvider
from .engine import ActiveLearningEngine, StepResult
from .state.store import ProjectState, StateStore
from .strategies.base import SamplingStrategy


class ActiveLearningProject:
    """
    Public entrypoint for users of the SDK.

    Think of this class as a "nice API wrapper" around the real engine.
    It keeps the surface area simple and stable, while the implementation
    lives in `ActiveLearningEngine`.

    Common flow:
    1. `project = ActiveLearningProject(name, workdir=...)`
    2. `project.configure(...)`
    3. `project.run(...)` or repeatedly call `project.run_step(...)`

    Attributes:
        _engine (ActiveLearningEngine):
            The real implementation object.
            Where: every method in this facade delegates to it.
            What: instance of `ActiveLearningEngine`.
            Why: keeps the user API small and stable while engine can grow internally.
    """

    def __init__(
        self,
        project_name: str,
        workdir: Union[str, Path],
        *,
        state_store: Optional[StateStore] = None,
        lock: bool = True,
    ) -> None:
        self._engine = ActiveLearningEngine(project_name, workdir, state_store=state_store, lock=lock)

    def __enter__(self) -> "ActiveLearningProject":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def project_name(self) -> str:
        return self._engine.project_name

    @property
    def workdir(self) -> Path:
        return self._engine.workdir

    def configure(
        self,
        *,
        dataset: Union[DatasetProvider, Any, str, Path],
        model: TextClassificationAdapter,
        label_schema: LabelSchema,
        label_backend_config: LabelBackendConfig,
        scheduler_config: SchedulerConfig,
        label_backend: Optional[LabelBackend] = None,
        annotation_policy: AnnotationPolicy = AnnotationPolicy(),
        cache_config: CacheConfig = CacheConfig(),
        fingerprint_config: FingerprintConfig = FingerprintConfig(),
        split_config: SplitConfig = SplitConfig(),
        prelabel_config: PrelabelConfig = PrelabelConfig(),
    ) -> None:
        """
        Configure the project and persist configuration to disk.

        This method is what "wires together" the pieces of the SDK:
        - dataset (where samples come from)
        - model adapter (how to predict/train/evaluate)
        - scheduler + strategy (how to pick next samples)
        - labeling backend (where humans/labelers annotate)
        - policies (how to resolve multiple annotations)

        Important:
        - Configuration is written into `workdir/state.json`.
        - The SDK uses that state to resume after crashes and avoid duplicates.
        """
        self._engine.configure(
            dataset=dataset,
            model=model,
            label_schema=label_schema,
            label_backend_config=label_backend_config,
            scheduler_config=scheduler_config,
            label_backend=label_backend,
            annotation_policy=annotation_policy,
            cache_config=cache_config,
            fingerprint_config=fingerprint_config,
            split_config=split_config,
            prelabel_config=prelabel_config,
        )

    def attach_runtime(
        self,
        *,
        dataset: Union[DatasetProvider, Any, str, Path],
        model: TextClassificationAdapter,
        label_backend: Optional[LabelBackend] = None,
    ) -> None:
        """
        Attach runtime objects to an already-configured project.

        Use this when you re-open an existing `workdir` in a new Python process.
        The config is loaded from `state.json`, but live Python objects (dataset/model/backend)
        must be provided again.
        """
        self._engine.attach_runtime(dataset=dataset, model=model, label_backend=label_backend)

    def register_strategy(self, strategy: SamplingStrategy) -> None:
        """
        Register a custom sampling strategy at runtime.

        After registration, `SchedulerConfig(strategy=...)` can refer to `strategy.name`.
        """
        self._engine.register_strategy(strategy)

    def run(
        self,
        *,
        budget: Optional[int] = None,
        batch_size: int = 50,
        stop_criteria: StopCriteria = StopCriteria(),
        resume: bool = True,
        poll_interval_seconds: int = 10,
    ) -> None:
        """
        Run the active-learning loop in a blocking way.

        Internally this repeatedly calls `run_step()` until stop criteria is reached.
        """
        self._engine.run(
            budget=budget,
            batch_size=batch_size,
            stop_criteria=stop_criteria,
            resume=resume,
            poll_interval_seconds=poll_interval_seconds,
        )

    def run_step(self, *, batch_size: int = 50, poll_interval_seconds: int = 0) -> StepResult:
        """
        Execute exactly one state-machine step and checkpoint state.

        This is the best API for notebooks/services where you want control over
        scheduling and retries.
        """
        return self._engine.run_step(batch_size=batch_size, poll_interval_seconds=poll_interval_seconds)

    def status(self) -> Dict[str, Any]:
        """Return a small status snapshot (counts, last metrics, active round)."""
        return self._engine.status()

    def get_state(self) -> ProjectState:
        """Return the full in-memory `ProjectState` object."""
        return self._engine.get_state()

    def list_rounds(self) -> list[Dict[str, Any]]:
        """Return short summaries for all rounds stored in state."""
        return self._engine.list_rounds()

    def get_round(self, round_id: str) -> Dict[str, Any]:
        return self._engine.get_round(round_id)

    def validate(self) -> Dict[str, Any]:
        return self._engine.validate()

    def generate_report(self, output_path: Union[str, Path] = "report.html") -> None:
        self._engine.generate_report(output_path=output_path)

    def export_labels(self, output_path: Union[str, Path], *, format: str = "jsonl") -> None:
        self._engine.export_labels(output_path=output_path, format=format)

    def export_dataset_split(self, output_dir: Union[str, Path], *, which: str = "labeled", format: str = "jsonl") -> None:
        self._engine.export_dataset_split(output_dir=output_dir, which=which, format=format)

    def cache_stats(self) -> Dict[str, Any]:
        return self._engine.cache_stats()

    def clear_cache(self, *, kind: str = "all") -> None:
        self._engine.clear_cache(kind=kind)

    def close(self) -> None:
        self._engine.close()
