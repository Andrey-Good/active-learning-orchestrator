from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    AnnotationPolicy,
    ConfigurationError,
    LabelBackendConfig,
    LabelSchema,
    ModelAdapterError,
    SchedulerConfig,
    SplitConfig,
    StateCorruptedError,
    StopCriteria,
)
from active_learning_sdk.backends.simulator import SimulatorLabelBackend
from active_learning_sdk.types import DataSample


REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyProvider:
    def __init__(self) -> None:
        self.rows = {
            "s0": "seed positive",
            "s1": "seed negative",
            "s2": "candidate one",
            "s3": "candidate two",
            "s4": "validation positive",
            "s5": "test negative",
        }

    def iter_sample_ids(self):
        yield from self.rows

    def get_sample(self, sample_id: str) -> DataSample:
        return DataSample(sample_id=sample_id, data={"text": self.rows[sample_id]})

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class TinyModel:
    def __init__(self, metrics: Mapping[str, Any] | None = None) -> None:
        self.metrics = dict(metrics or {"accuracy": 1.0})

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> Mapping[str, Any]:
        del texts, labels
        return self.metrics

    def get_model_id(self) -> str:
        return f"tiny-model:{sorted(self.metrics)}"


def _configure_project(
    workdir: Path,
    *,
    model: TinyModel | None = None,
    explicit_splits: Mapping[str, Sequence[str]] | None = None,
    lock: bool = False,
) -> ActiveLearningProject:
    project = ActiveLearningProject("blackbox-2026-04-30", workdir, lock=lock)
    project.configure(
        dataset=TinyProvider(),
        model=model or TinyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="simulator"),
        label_backend=SimulatorLabelBackend(label_by_sample_id={"s2": "positive", "s3": "negative"}),
        scheduler_config=SchedulerConfig(mode="single", strategy="random"),
        annotation_policy=AnnotationPolicy(mode="latest", min_votes=1),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits=explicit_splits
            or {"train": ["s0", "s1", "s2", "s3"], "val": ["s4"], "test": ["s5"]},
        ),
    )
    return project


def test_reconfigure_rejects_changed_explicit_split_identity_before_labels(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, lock=True)
    project.close()
    original_state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    original_split_config = original_state["split_config"]

    reopened = ActiveLearningProject("blackbox-2026-04-30", tmp_path, lock=True)
    with pytest.raises(ConfigurationError, match="split_config"):
        reopened.configure(
            dataset=TinyProvider(),
            model=TinyModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="simulator"),
            label_backend=SimulatorLabelBackend(label_by_sample_id={"s2": "positive", "s3": "negative"}),
            scheduler_config=SchedulerConfig(mode="single", strategy="random"),
            annotation_policy=AnnotationPolicy(mode="latest", min_votes=1),
            split_config=SplitConfig(
                mode="explicit",
                explicit_splits={"train": ["s0", "s1", "s2"], "val": ["s3"], "test": ["s4", "s5"]},
            ),
        )
    reopened.close()

    persisted_state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert persisted_state["split_config"] == original_split_config


def test_non_numeric_evaluate_metrics_raise_model_adapter_error_without_persisting(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=TinyModel({"accuracy": "perfect"}),
        explicit_splits={"train": ["s0", "s2", "s3"], "val": ["s1"], "test": ["s4", "s5"]},
    )
    project.import_labels({"s0": "positive", "s1": "negative"}, source="seed")

    with pytest.raises(ModelAdapterError, match="accuracy"):
        project.run(batch_size=1, stop_criteria=StopCriteria(max_rounds=1), poll_interval_seconds=0)

    assert project.status()["last_metrics"] == {}
    assert project.get_state().metrics_history == []


@pytest.mark.parametrize("entrypoint", ["run", "run_step"])
@pytest.mark.parametrize("batch_size", [0, -1])
def test_non_positive_batch_size_is_rejected_before_round_creation(
    tmp_path: Path,
    entrypoint: str,
    batch_size: int,
) -> None:
    project = _configure_project(tmp_path / f"{entrypoint}-{batch_size}")
    project.import_labels({"s0": "positive", "s1": "negative"}, source="seed")
    before_rounds = project.list_rounds()

    with pytest.raises(ConfigurationError, match="batch_size"):
        if entrypoint == "run":
            project.run(batch_size=batch_size, stop_criteria=StopCriteria(max_rounds=1), poll_interval_seconds=0)
        else:
            project.run_step(batch_size=batch_size, poll_interval_seconds=0)

    assert project.list_rounds() == before_rounds


def test_corrupted_state_open_releases_same_process_lock(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, lock=True)
    project.close()
    state_path = tmp_path / "state.json"
    valid_state = state_path.read_text(encoding="utf-8")
    state_path.write_text("{ broken json", encoding="utf-8")

    with pytest.raises(StateCorruptedError):
        ActiveLearningProject("blackbox-2026-04-30", tmp_path, lock=True)

    state_path.write_text(valid_state, encoding="utf-8")
    reopened = ActiveLearningProject("blackbox-2026-04-30", tmp_path, lock=True)
    reopened.close()


@pytest.mark.parametrize(
    "script,args",
    [
        (
            "sdk_first_benchmark.py",
            ["--datasets", "separable_topics", "--strategies", "random", "--budgets", "12", "--seeds", "13"],
        ),
        ("reference_strategy_benchmark.py", ["--preset", "smoke"]),
    ],
)
def test_benchmark_nonempty_output_dir_refusal_has_no_traceback(
    tmp_path: Path,
    script: str,
    args: list[str],
) -> None:
    output_dir = tmp_path / script
    output_dir.mkdir()
    (output_dir / "sentinel.txt").write_text("stale", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / script),
            *args,
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Refusing to write benchmark artifacts into non-empty output directory" in combined
    assert "Traceback" not in combined


def test_sdk_first_benchmark_overwrite_removes_stale_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "sdk-overwrite"
    output_dir.mkdir()
    stale_file = output_dir / "sentinel.txt"
    stale_file.write_text("stale", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "sdk_first_benchmark.py"),
            "--datasets",
            "separable_topics",
            "--strategies",
            "random",
            "--budgets",
            "12",
            "--seeds",
            "13",
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not stale_file.exists()
    assert (output_dir / "manifest.json").exists()
