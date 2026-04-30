from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import native_external_benchmark as native_benchmark


def _read_result_rows(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "native_external_results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_missing_optional_libraries_write_skipped_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_import(module_name: str) -> Any:
        raise native_benchmark.NativeBenchmarkSkip(f"{module_name} intentionally unavailable")

    monkeypatch.setattr(native_benchmark, "_import_optional_module", fake_import)
    output_dir = tmp_path / "native_external"

    assert native_benchmark.main(["--preset", "smoke", "--output-dir", str(output_dir)]) == 0

    rows = _read_result_rows(output_dir)
    assert len(rows) == 4
    assert {row["status"] for row in rows} == {"skipped"}
    assert all(row["claim_category"] == native_benchmark.CLAIM_CATEGORY for row in rows)
    assert all("intentionally unavailable" in row["skip_reason"] for row in rows)
    summary = json.loads((output_dir / "native_external_summary.json").read_text(encoding="utf-8"))
    assert summary["claim_category"] == native_benchmark.CLAIM_CATEGORY
    assert summary["row_counts"]["skipped"] == 4
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["claim_category"] == native_benchmark.CLAIM_CATEGORY
    assert manifest["benchmark_claim_category"] == native_benchmark.CLAIM_CATEGORY
    assert manifest["artifacts"]["results_csv"] == "native_external_results.csv"
    assert "native_external_results.csv" in manifest["artifact_names"]


def test_modal_fake_module_native_functions_are_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, int, int]] = []
    modAL_module = types.ModuleType("modAL")
    uncertainty_module = types.ModuleType("modAL.uncertainty")

    def make_sampling_function(name: str) -> Any:
        def sampling_function(classifier: Any, x_pool: Any, n_instances: int) -> tuple[list[int], None]:
            probabilities = classifier.predict_proba(x_pool)
            calls.append((name, int(x_pool.shape[0]), len(probabilities)))
            assert n_instances == native_benchmark.SMOKE_BATCH_SIZE
            return [2, 0], None

        return sampling_function

    uncertainty_module.entropy_sampling = make_sampling_function("entropy_sampling")  # type: ignore[attr-defined]
    uncertainty_module.margin_sampling = make_sampling_function("margin_sampling")  # type: ignore[attr-defined]
    uncertainty_module.uncertainty_sampling = make_sampling_function("uncertainty_sampling")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modAL", modAL_module)
    monkeypatch.setitem(sys.modules, "modAL.uncertainty", uncertainty_module)
    output_dir = tmp_path / "modal_native"

    assert (
        native_benchmark.main(
            [
                "--preset",
                "smoke",
                "--libraries",
                "modal",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    rows = _read_result_rows(output_dir)
    assert {row["strategy"] for row in rows} == {
        "modal_native_entropy",
        "modal_native_margin",
        "modal_native_least_confidence",
    }
    assert {row["status"] for row in rows} == {"ok"}
    assert {call[0] for call in calls} == {
        "entropy_sampling",
        "margin_sampling",
        "uncertainty_sampling",
    }
    assert all(pool_rows == probability_rows for _, pool_rows, probability_rows in calls)


def test_skactiveml_fake_module_query_class_is_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    skactiveml_module = types.ModuleType("skactiveml")
    pool_module = types.ModuleType("skactiveml.pool")
    utils_module = types.ModuleType("skactiveml.utils")
    utils_module.MISSING_LABEL = "MISSING"  # type: ignore[attr-defined]

    class FakeUncertaintySampling:
        def __init__(self, method: str = "least_confident") -> None:
            self.method = method

        def query(
            self,
            *,
            X: Any,
            y: list[Any],
            clf: Any,
            batch_size: int,
            return_utilities: bool = False,
        ) -> list[int]:
            probabilities = clf.predict_proba(X)
            calls.append(
                {
                    "method": self.method,
                    "x_rows": int(X.shape[0]),
                    "y_rows": len(y),
                    "probability_rows": len(probabilities),
                    "batch_size": batch_size,
                    "return_utilities": return_utilities,
                    "missing_labels": set(y),
                }
            )
            return [1, 0]

    pool_module.UncertaintySampling = FakeUncertaintySampling  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "skactiveml", skactiveml_module)
    monkeypatch.setitem(sys.modules, "skactiveml.pool", pool_module)
    monkeypatch.setitem(sys.modules, "skactiveml.utils", utils_module)
    output_dir = tmp_path / "skactiveml_native"

    assert (
        native_benchmark.main(
            [
                "--preset",
                "smoke",
                "--libraries",
                "skactiveml",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    rows = _read_result_rows(output_dir)
    assert len(rows) == 1
    assert rows[0]["strategy"] == "skactiveml_native_uncertainty"
    assert rows[0]["status"] == "ok"
    assert calls == [
        {
            "method": "entropy",
            "x_rows": calls[0]["x_rows"],
            "y_rows": calls[0]["x_rows"],
            "probability_rows": calls[0]["x_rows"],
            "batch_size": native_benchmark.SMOKE_BATCH_SIZE,
            "return_utilities": False,
            "missing_labels": {"MISSING"},
        }
    ]


def test_native_external_refuses_non_empty_output_dir_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "native_external"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        native_benchmark.main(["--preset", "smoke", "--output-dir", str(output_dir)])

    assert sentinel.read_text(encoding="utf-8") == "do not replace"
