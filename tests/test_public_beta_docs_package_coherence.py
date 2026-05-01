from __future__ import annotations

import tomllib
from pathlib import Path

from active_learning_sdk import (
    ActiveLearningProject,
    AnnotationPolicy,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
    StopCriteria,
)
from active_learning_sdk.backends.simulator import SimulatorLabelBackend
from active_learning_sdk.types import DataSample


REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyProvider:
    def __init__(self) -> None:
        self.rows = {
            "s1": "free trial works",
            "s2": "invoice failed",
            "s3": "upgrade account",
            "s4": "refund request",
        }

    def iter_sample_ids(self):
        return iter(self.rows)

    def get_sample(self, sample_id: str) -> DataSample:
        return DataSample(sample_id=sample_id, data={"text": self.rows[sample_id]})

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class TinyModel:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def predict_proba(self, texts, batch_size: int = 32):
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts, labels, **kwargs) -> None:
        self.labels = list(labels)

    def evaluate(self, texts, labels) -> dict[str, float]:
        return {"accuracy": 1.0 if labels else 0.0}

    def get_model_id(self) -> str:
        return f"tiny-model-{len(self.labels)}"


def test_readme_core_simulator_quickstart_smokes_without_optional_dependencies(tmp_path: Path) -> None:
    project = ActiveLearningProject("quickstart", workdir=tmp_path / "quickstart")
    project.configure(
        dataset=TinyProvider(),
        model=TinyModel(),
        label_schema=LabelSchema(
            task="text_classification",
            labels=["negative", "positive"],
        ),
        label_backend_config=LabelBackendConfig(backend="simulator"),
        label_backend=SimulatorLabelBackend(
            label_by_sample_id={"s3": "positive", "s4": "negative"}
        ),
        scheduler_config=SchedulerConfig(mode="single", strategy="random"),
        annotation_policy=AnnotationPolicy(mode="latest", min_votes=1),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={
                "train": ["s1", "s2", "s3", "s4"],
                "val": [],
                "test": [],
            },
        ),
    )

    project.import_labels({"s1": "positive", "s2": "negative"}, source="seed")
    project.run(
        batch_size=2,
        stop_criteria=StopCriteria(max_rounds=1),
        poll_interval_seconds=0,
    )

    assert project.status()["counts"] == {
        "labeled": 4,
        "unlabeled": 0,
        "needs_review": 0,
        "invalid": 0,
    }
    project.close()


def test_public_beta_docs_and_package_metadata_are_coherent() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    docs_readme = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    benchmark_readme = (REPO_ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    optional_deps = pyproject["project"]["optional-dependencies"]
    assert set(optional_deps) == {
        "sklearn",
        "huggingface",
        "datasets",
        "xxhash",
        "benchmarks",
        "all",
    }
    assert "Development Status :: 4 - Beta" in pyproject["project"]["classifiers"]

    for extra in optional_deps:
        assert f"`{extra}`" in readme
        assert f"`{extra}`" in docs_readme

    assert "## Core Simulator Quickstart" in readme
    assert "LabelBackendConfig(backend=\"simulator\")" in readme
    assert "does not require `pandas`, optional extras, or a live Label Studio service" in readme
    assert "External Mode" in readme
    assert "Managed Docker Mode" in readme
    assert "requires explicit credentials" in readme
    assert "active-learning@example.local" not in readme
    assert "native_external_library_workflow_smoke" in benchmark_readme
    assert "native_external_workflow_smoke" not in benchmark_readme


def test_repo_managed_compose_uses_explicit_credential_contract() -> None:
    compose = (REPO_ROOT / "docker" / "label_studio" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert "${LABEL_STUDIO_USERNAME:?LABEL_STUDIO_USERNAME is required}" in compose
    assert "${LABEL_STUDIO_PASSWORD:?LABEL_STUDIO_PASSWORD is required}" in compose
    assert "${LABEL_STUDIO_USER_TOKEN:?LABEL_STUDIO_USER_TOKEN is required}" in compose
    assert "active-learning@example.local" not in compose
    assert "active-learning-sdk-token" not in compose
