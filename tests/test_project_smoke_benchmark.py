from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import sdk_first_benchmark


def test_project_smoke_uses_public_sklearn_adapter_and_records_public_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "project_smoke"
    sdk_first_benchmark.ensure_benchmark_dependencies()

    sdk_first_benchmark.run_project_smoke(
        output_dir=output_dir,
        seed=13,
        initial_seed_size=9,
        batch_size=6,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["model_adapter"]["source"] == "active_learning_sdk.adapters"
    assert summary["model_adapter"]["class"] == "SklearnTextClassifierAdapter"
    assert (
        summary["model_adapter"]["qualified_class"]
        == "active_learning_sdk.adapters.sklearn.SklearnTextClassifierAdapter"
    )
    assert summary["model_adapter"]["estimator_class"] == "sklearn.pipeline.Pipeline"
    assert summary["steps"][0]["step"] == "train_eval"
    assert summary["steps"][0]["details"]["seed"] is True
    assert summary["validation"]["public_step_sequence"]["active_round_step_sequence"] == [
        "select",
        "push",
        "wait",
        "pull",
        "train_eval",
        "update",
    ]
    assert summary["validation"]["model_id_changed_after_public_steps"] is True
    assert summary["validation"]["active_round_has_metrics_after_training"] is True
    assert [record["step"] for record in summary["validation"]["metrics_history"][:2]] == ["seed_eval", "eval"]
    assert all(record["metrics"] for record in summary["validation"]["metrics_history"][:2])
    assert summary["validation"]["completed_active_rounds"] == 1
    assert summary["private_state_mutation_used"] is False
    assert summary["model_warm_started_from_imported_seed"] is False
    assert "model_fit_count_after_public_steps" not in summary
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_schema_version"] == sdk_first_benchmark.BENCHMARK_ARTIFACT_SCHEMA_VERSION
    assert manifest["preset"] == "project_smoke"
    assert manifest["artifacts"]["workdir_state"] == "workdir/state.json"


def test_project_smoke_refuses_non_empty_output_dir_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "project_smoke"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        sdk_first_benchmark.run_project_smoke(
            output_dir=output_dir,
            seed=13,
            initial_seed_size=9,
            batch_size=6,
        )

    assert sentinel.read_text(encoding="utf-8") == "do not replace"
