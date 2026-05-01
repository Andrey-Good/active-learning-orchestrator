from __future__ import annotations

import os
import shutil

import pytest

from active_learning_sdk.backends.label_studio import LabelStudioBackend
from active_learning_sdk.configs import LabelBackendConfig, LabelSchema
from active_learning_sdk.exceptions import ActiveLearningError, InfrastructureError
from active_learning_sdk.types import DataSample

def test_live_managed_label_studio_smoke_roundtrip() -> None:
    if os.environ.get("ACTIVE_LEARNING_RUN_LIVE_LABEL_STUDIO") != "1":
        pytest.skip("Set ACTIVE_LEARNING_RUN_LIVE_LABEL_STUDIO=1 to run live managed Label Studio tests.")
    if shutil.which("docker") is None and shutil.which("docker-compose") is None:
        pytest.skip("Docker/Docker Compose is not available on PATH.")
    missing = [
        name
        for name in (
            "ACTIVE_LEARNING_LABEL_STUDIO_USERNAME",
            "ACTIVE_LEARNING_LABEL_STUDIO_PASSWORD",
            "ACTIVE_LEARNING_LABEL_STUDIO_TOKEN",
        )
        if not os.environ.get(name)
    ]
    if missing:
        pytest.skip(f"Live Label Studio test requires env vars: {', '.join(missing)}")

    backend = LabelStudioBackend(
        LabelBackendConfig(
            backend="label_studio",
            mode="managed_docker",
            managed_port=int(os.environ.get("ACTIVE_LEARNING_LABEL_STUDIO_PORT", "9091")),
        )
    )
    try:
        backend.ensure_ready(LabelSchema(task="text_classification", labels=["negative", "positive"]))
        pushed = backend.push_round(
            "live_smoke",
            [DataSample(sample_id="s1", data={"text": "live smoke sample"})],
            prelabels={"s1": "positive"},
        )
        assert pushed.task_ids == {"s1": pushed.task_ids["s1"]}
    except (InfrastructureError, ActiveLearningError) as error:
        pytest.skip(f"Live Label Studio environment is unavailable: {error}")
    finally:
        backend.close()
