from __future__ import annotations

import json
from pathlib import Path

import pytest

from active_learning_sdk.engine import ActiveLearningEngine
from active_learning_sdk.exceptions import StateCorruptedError
from active_learning_sdk.state.store import JsonFileStateStore, ProjectState, validate_state_version
from active_learning_sdk.types import MetricRecord


def test_state_store_rejects_non_finite_float_before_writing(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"existing": true}\n', encoding="utf-8")
    state = ProjectState(
        state_version=1,
        project_name="strict-state",
        created_at=1.0,
        updated_at=2.0,
        metrics_history=[
            MetricRecord(
                step="eval",
                created_at=3.0,
                metrics={"accuracy": float("nan")},
            )
        ],
    )

    with pytest.raises(StateCorruptedError, match=r"non-finite float.*metrics_history\[0\]\.metrics\.accuracy"):
        JsonFileStateStore(state_path).save_atomic(state)

    assert state_path.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_state_store_writes_strict_json_for_finite_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = ProjectState(
        state_version=1,
        project_name="strict-state",
        created_at=1.0,
        updated_at=2.0,
        scheduler_config={"strategy": "random", "mode": "single"},
    )

    JsonFileStateStore(state_path).save_atomic(state)

    raw = state_path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    parsed = json.loads(raw)
    assert parsed["project_name"] == "strict-state"
    json.dumps(parsed, allow_nan=False)


def test_state_store_rejects_non_finite_constants_on_load(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        """
{
  "state_version": 1,
  "project_name": "strict-state",
  "created_at": 1.0,
  "updated_at": NaN
}
""",
        encoding="utf-8",
    )

    with pytest.raises(StateCorruptedError, match="non-finite JSON constant"):
        JsonFileStateStore(state_path).load()


@pytest.mark.parametrize("overflow_literal", ["1e999", "-1e999"])
def test_state_store_rejects_overflowed_json_numbers_on_load(
    tmp_path: Path,
    overflow_literal: str,
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        f"""
{{
  "state_version": 1,
  "project_name": "strict-state",
  "created_at": {overflow_literal},
  "updated_at": 2.0
}}
""",
        encoding="utf-8",
    )

    with pytest.raises(StateCorruptedError, match=r"non-finite float.*created_at"):
        JsonFileStateStore(state_path).load()


def test_unsupported_state_version_error_is_clear(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "state_version": 999,
                "project_name": "strict-state",
                "created_at": 1.0,
                "updated_at": 2.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StateCorruptedError, match=r"Unsupported state_version 999.*Supported versions: 1"):
        ActiveLearningEngine("strict-state", tmp_path, lock=False)


def test_state_version_validation_hook_accepts_current_version() -> None:
    assert validate_state_version(1) == 1
