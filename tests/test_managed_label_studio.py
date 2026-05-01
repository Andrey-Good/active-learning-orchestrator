from __future__ import annotations

import subprocess
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from active_learning_sdk.backends import managed_docker as managed_docker_module
from active_learning_sdk.backends.managed_docker import ManagedLabelStudioRuntime
from active_learning_sdk.configs import LabelBackendConfig
from active_learning_sdk.exceptions import InfrastructureError


def _config(**overrides: Any) -> LabelBackendConfig:
    values: dict[str, Any] = {
        "backend": "label_studio",
        "mode": "managed_docker",
        "managed_port": 9091,
    }
    values.update(overrides)
    return LabelBackendConfig(**values)


def test_managed_runtime_copies_packaged_default_assets_and_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACTIVE_LEARNING_LABEL_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("ACTIVE_LEARNING_LABEL_STUDIO_USERNAME", "user@example.test")
    monkeypatch.setenv("ACTIVE_LEARNING_LABEL_STUDIO_PASSWORD", "secret")

    runtime = ManagedLabelStudioRuntime(_config(api_token="token-123"))

    assert runtime.compose_path == tmp_path / "docker-compose.yml"
    assert runtime.compose_path.exists()
    assert (tmp_path / "nginx.conf").exists()
    assert (tmp_path / "data").is_dir()
    assert runtime.resolved_url == "http://127.0.0.1:9091"
    assert runtime.compose_env() == {
        "LABEL_STUDIO_HOST_PORT": "9091",
        "LABEL_STUDIO_USERNAME": "user@example.test",
        "LABEL_STUDIO_PASSWORD": "secret",
        "LABEL_STUDIO_USER_TOKEN": "token-123",
    }

    packaged_root = resources.files("active_learning_sdk.backends.assets").joinpath("label_studio")
    assert runtime.compose_path.read_bytes() == packaged_root.joinpath("docker-compose.yml").read_bytes()
    assert (tmp_path / "nginx.conf").read_bytes() == packaged_root.joinpath("nginx.conf").read_bytes()


def test_packaged_and_repo_managed_assets_are_in_sync() -> None:
    packaged_root = resources.files("active_learning_sdk.backends.assets").joinpath("label_studio")
    repo_root = Path(__file__).resolve().parents[1] / "docker" / "label_studio"

    for filename in ("docker-compose.yml", "nginx.conf"):
        assert packaged_root.joinpath(filename).read_bytes() == (repo_root / filename).read_bytes()


def test_compose_project_command_uses_default_and_custom_compose_names(tmp_path: Path) -> None:
    default_file = tmp_path / "docker-compose.yml"
    default_file.write_text("services: {}\n", encoding="utf-8")
    default_runtime = ManagedLabelStudioRuntime(_config(managed_compose_path=str(default_file)))
    default_runtime._compose_cmd = ["docker", "compose"]

    assert default_runtime.compose_dir == tmp_path
    assert default_runtime.compose_project_command() == ["docker", "compose"]

    custom_file = tmp_path / "custom-compose.yml"
    custom_file.write_text("services: {}\n", encoding="utf-8")
    custom_runtime = ManagedLabelStudioRuntime(_config(managed_compose_path=str(custom_file)))
    custom_runtime._compose_cmd = ["docker", "compose"]

    assert custom_runtime.compose_dir == tmp_path
    assert custom_runtime.compose_project_command() == ["docker", "compose", "-f", "custom-compose.yml"]
    assert custom_runtime.compose_config_command() == ["docker", "compose", "-f", "custom-compose.yml", "config"]


def test_ensure_running_exports_env_without_real_docker_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    runtime = ManagedLabelStudioRuntime(_config(managed_compose_path=str(compose_file), api_token="token"))
    runtime._compose_cmd = ["docker", "compose"]
    calls: list[dict[str, Any]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(managed_docker_module.subprocess, "run", fake_run)

    runtime.ensure_running()

    assert calls == [
        {
            "command": ["docker", "compose", "up", "-d"],
            "check": True,
            "capture_output": True,
            "text": True,
            "cwd": tmp_path,
            "env": calls[0]["env"],
            "timeout": 120,
        }
    ]
    assert calls[0]["env"]["LABEL_STUDIO_HOST_PORT"] == "9091"
    assert calls[0]["env"]["LABEL_STUDIO_USER_TOKEN"] == "token"


def test_ensure_running_failure_includes_redacted_compose_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    runtime = ManagedLabelStudioRuntime(_config(managed_compose_path=str(compose_file), api_token="token-secret"))
    runtime._compose_cmd = ["docker", "compose"]
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[-2:] == ["up", "-d"]:
            raise subprocess.CalledProcessError(1, command, stderr="failed token-secret password=bad")
        if command[-1:] == ["ps"]:
            return subprocess.CompletedProcess(command, 0, stdout="label-studio unhealthy token-secret", stderr="")
        if "logs" in command:
            return subprocess.CompletedProcess(command, 0, stdout="password=bad\nready=false", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(managed_docker_module.subprocess, "run", fake_run)

    with pytest.raises(InfrastructureError) as error:
        runtime.ensure_running()

    message = str(error.value)
    assert "compose_ps" in message
    assert "compose_logs" in message
    assert "token-secret" not in message
    assert "password=bad" not in message
    assert "<redacted>" in message
    assert ["docker", "compose", "ps"] in calls
    assert ["docker", "compose", "logs", "--tail", "80"] in calls


def test_compose_command_reports_clear_error_when_compose_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    runtime = ManagedLabelStudioRuntime(_config(managed_compose_path=str(compose_file)))

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(managed_docker_module.subprocess, "run", fake_run)

    with pytest.raises(InfrastructureError, match="requires Docker Compose"):
        runtime.compose_command()


def test_missing_managed_assets_report_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACTIVE_LEARNING_LABEL_STUDIO_HOME", str(tmp_path / "runtime"))
    monkeypatch.setattr(ManagedLabelStudioRuntime, "_sync_packaged_assets", lambda self, runtime_dir: False)
    monkeypatch.setattr(ManagedLabelStudioRuntime, "_resolve_repo_asset_dir", lambda self: tmp_path / "missing")

    with pytest.raises(InfrastructureError, match="Managed Docker asset is missing"):
        ManagedLabelStudioRuntime(_config())
