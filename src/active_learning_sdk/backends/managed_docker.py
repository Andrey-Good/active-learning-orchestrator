"""
Managed Docker helpers for local Label Studio instances.
"""

from __future__ import annotations


import os
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List

from ..configs import LabelBackendConfig
from ..exceptions import InfrastructureError

_RUNTIME_HOME_ENV = "ACTIVE_LEARNING_LABEL_STUDIO_HOME"
_USERNAME_ENV = "ACTIVE_LEARNING_LABEL_STUDIO_USERNAME"
_PASSWORD_ENV = "ACTIVE_LEARNING_LABEL_STUDIO_PASSWORD"
_TOKEN_ENV = "ACTIVE_LEARNING_LABEL_STUDIO_TOKEN"
_SECRET_ENV_NAMES = (_USERNAME_ENV, _PASSWORD_ENV, _TOKEN_ENV, "LABEL_STUDIO_USER_TOKEN", "LABEL_STUDIO_PASSWORD")
_DIAGNOSTIC_TIMEOUT_SECONDS = 15
_DIAGNOSTIC_TEXT_LIMIT = 8000


class ManagedLabelStudioRuntime:
    """
    Small wrapper around docker compose for local Label Studio startup.
    """

    def __init__(self, config: LabelBackendConfig) -> None:
        self._config = config
        self._uses_packaged_compose = config.managed_compose_path is None
        self.compose_path = self._resolve_compose_path()
        self.username = os.environ.get(_USERNAME_ENV)
        self.password = os.environ.get(_PASSWORD_ENV)
        self._token = config.api_token or os.environ.get(_TOKEN_ENV)
        if self._uses_packaged_compose:
            self._require_packaged_credentials()
        self._compose_cmd: List[str] | None = None

    @property
    def token(self) -> str:
        return self._require_token()

    @property
    def resolved_url(self) -> str:
        if self._config.url:
            return self._config.url.rstrip("/")
        return f"http://127.0.0.1:{self._config.managed_port}"

    def compose_env(self) -> Dict[str, str]:
        if self._uses_packaged_compose:
            self._require_packaged_credentials()

        env = {
            "LABEL_STUDIO_HOST_PORT": str(self._config.managed_port),
            "LABEL_STUDIO_USER_TOKEN": self._require_token(),
        }
        if self.username is not None:
            env["LABEL_STUDIO_USERNAME"] = self.username
        if self.password is not None:
            env["LABEL_STUDIO_PASSWORD"] = self.password
        return env

    def ensure_running(self) -> None:
        env = os.environ.copy()
        env.update(self.compose_env())
        try:
            subprocess.run(
                [*self.compose_project_command(), "up", "-d"],
                check=True,
                capture_output=True,
                text=True,
                cwd=self.compose_dir,
                env=env,
                timeout=120,
            )
        except FileNotFoundError as error:
            raise InfrastructureError("Docker is not installed or not available on PATH.") from error
        except subprocess.CalledProcessError as error:
            stderr = (error.stderr or "").strip()
            raise InfrastructureError(
                "Failed to start managed Label Studio with docker compose. "
                f"stderr={self._redact_text(stderr)} diagnostics={self.diagnostics()}".strip()
            ) from error
        except subprocess.TimeoutExpired as error:
            raise InfrastructureError(
                f"Docker Compose startup timed out for managed Label Studio. diagnostics={self.diagnostics()}"
            ) from error

    def compose_command(self) -> List[str]:
        if self._compose_cmd is not None:
            return list(self._compose_cmd)
        for command in (["docker", "compose"], ["docker-compose"]):
            try:
                subprocess.run(
                    [*command, "version"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self._compose_cmd = command
                return list(command)
            except FileNotFoundError:
                continue
            except subprocess.CalledProcessError:
                continue
            except subprocess.TimeoutExpired as error:
                raise InfrastructureError("Docker Compose version probe timed out.") from error
        raise InfrastructureError(
            "Managed Docker mode requires Docker Compose. Install Docker Desktop or docker compose v2."
        )

    def compose_config_command(self) -> List[str]:
        return [*self.compose_project_command(), "config"]

    def diagnostics(self, *, log_tail: int = 80) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "compose_path": str(self.compose_path),
            "compose_dir": str(self.compose_dir),
            "resolved_url": self.resolved_url,
            "uses_packaged_compose": self._uses_packaged_compose,
            "managed_port": self._config.managed_port,
            "secret_env": {name: ("<set>" if os.environ.get(name) else "<unset>") for name in _SECRET_ENV_NAMES},
        }
        try:
            diagnostics["compose_command"] = self.compose_project_command()
        except Exception as error:
            diagnostics["compose_command_error"] = f"{type(error).__name__}: {error}"
            return diagnostics

        diagnostics["compose_ps"] = self._run_compose_diagnostic(["ps"])
        diagnostics["compose_logs"] = self._run_compose_diagnostic(["logs", "--tail", str(max(0, log_tail))])
        return diagnostics

    @property
    def compose_dir(self) -> Path:
        return self.compose_path.parent

    def compose_project_command(self) -> List[str]:
        command = self.compose_command()
        default_names = {"compose.yml", "compose.yaml", "docker-compose.yml", "docker-compose.yaml"}
        if self.compose_path.name in default_names:
            return command
        return [*command, "-f", self.compose_path.name]

    def _resolve_compose_path(self) -> Path:
        if self._config.managed_compose_path:
            path = Path(self._config.managed_compose_path).expanduser().resolve()
            if not path.exists():
                raise InfrastructureError(f"Managed compose file does not exist: {path}")
            return path

        runtime_dir = self._prepare_runtime_project()
        return runtime_dir / "docker-compose.yml"

    def _prepare_runtime_project(self) -> Path:
        runtime_dir = self._resolve_runtime_home()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "data").mkdir(parents=True, exist_ok=True)

        if self._sync_packaged_assets(runtime_dir):
            return runtime_dir

        asset_dir = self._resolve_repo_asset_dir()
        self._sync_asset_dir(asset_dir, runtime_dir)
        return runtime_dir

    def _sync_packaged_assets(self, runtime_dir: Path) -> bool:
        try:
            assets_root = resources.files("active_learning_sdk.backends.assets")
            label_studio_root = assets_root.joinpath("label_studio")
            compose_resource = label_studio_root.joinpath("docker-compose.yml")
            nginx_resource = label_studio_root.joinpath("nginx.conf")
            if compose_resource.is_file() and nginx_resource.is_file():
                self._sync_traversable_file(compose_resource, runtime_dir / "docker-compose.yml")
                self._sync_traversable_file(nginx_resource, runtime_dir / "nginx.conf")
                return True
        except ModuleNotFoundError:
            return False
        return False

    def _resolve_repo_asset_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        compose_path = repo_root / "docker" / "label_studio" / "docker-compose.yml"
        if not compose_path.exists():
            raise InfrastructureError(
                "Managed Docker mode could not locate docker/label_studio/docker-compose.yml."
            )
        return compose_path.parent

    def _sync_asset_dir(self, asset_dir: Path, runtime_dir: Path) -> None:
        for filename in ("docker-compose.yml", "nginx.conf"):
            source = asset_dir / filename
            if not source.exists():
                raise InfrastructureError(f"Managed Docker asset is missing: {source}")
            destination = runtime_dir / filename
            if not destination.exists() or source.read_text(encoding="utf-8") != destination.read_text(encoding="utf-8"):
                shutil.copyfile(source, destination)

    def _sync_traversable_file(self, source: object, destination: Path) -> None:
        data = source.read_bytes()  # type: ignore[attr-defined]
        if destination.exists() and destination.read_bytes() == data:
            return
        destination.write_bytes(data)

    def _run_compose_diagnostic(self, args: List[str]) -> Dict[str, Any]:
        env = os.environ.copy()
        try:
            env.update(self.compose_env())
            result = subprocess.run(
                [*self.compose_project_command(), *args],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.compose_dir,
                env=env,
                timeout=_DIAGNOSTIC_TIMEOUT_SECONDS,
            )
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}
        return {
            "returncode": result.returncode,
            "stdout": self._redact_text(result.stdout or ""),
            "stderr": self._redact_text(result.stderr or ""),
        }

    def _redact_text(self, text: str) -> str:
        redacted = text
        for secret in (self.username, self.password, self._token, os.environ.get(_TOKEN_ENV)):
            if secret:
                redacted = redacted.replace(secret, "<redacted>")
        redacted = re.sub(
            r"(?i)(token|password|secret|authorization|credential)(['\"=:\s]+)([^'\"\s,}]+)",
            r"\1\2<redacted>",
            redacted,
        )
        if len(redacted) > _DIAGNOSTIC_TEXT_LIMIT:
            return redacted[:_DIAGNOSTIC_TEXT_LIMIT] + "...<truncated>"
        return redacted

    def _resolve_runtime_home(self) -> Path:
        configured = os.environ.get(_RUNTIME_HOME_ENV)
        if configured:
            return Path(configured).expanduser().resolve()
        return (Path.home() / ".active-learning-sdk" / "label_studio").resolve()

    def _require_token(self) -> str:
        if self._token:
            return self._token
        raise InfrastructureError(
            "Managed Label Studio requires an explicit token secret via "
            "LabelBackendConfig.api_token or ACTIVE_LEARNING_LABEL_STUDIO_TOKEN."
        )

    def _require_packaged_credentials(self) -> None:
        missing = []
        if not self.username:
            missing.append(_USERNAME_ENV)
        if not self.password:
            missing.append(_PASSWORD_ENV)
        if not self._token:
            missing.append(f"LabelBackendConfig.api_token or {_TOKEN_ENV}")
        if missing:
            raise InfrastructureError(
                "Managed Label Studio packaged runtime requires explicit credential secrets; "
                f"missing: {', '.join(missing)}."
            )
