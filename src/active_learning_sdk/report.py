"""
Dependency-free project reporting.

Reports are generated from persisted ProjectState only. The generator intentionally
does not require runtime dataset/model/backend objects and never mutates state.
"""

from __future__ import annotations


import dataclasses
import enum
import html
import importlib.metadata
import json
import math
import numbers
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Union

from .state.store import ProjectState
from .types import SampleStatus
from .utils import atomic_write_text, sha256_file


REPORT_ARTIFACT_SCHEMA_VERSION = 1
_PACKAGE_NAME = "active-learning-sdk"
_KNOWN_REPORT_FILENAMES = {
    ".json": "summary.json",
    ".md": "report.md",
    ".html": "report.html",
}


class ReportGenerator:
    """
    Generate strict JSON, Markdown, and HTML audit reports from ProjectState.

    The generated JSON payload is the source of truth. Markdown and HTML are
    rendered from that same sanitized payload so all artifacts agree.
    """

    def generate_report(
        self,
        state: ProjectState,
        output_path: Union[str, Path] = "report.html",
        *,
        workdir: Union[str, Path] | None = None,
        state_path: Union[str, Path] | None = None,
    ) -> Dict[str, Path]:
        """
        Write JSON, Markdown, and HTML report artifacts.

        Path behavior:
        - directory/no suffix: write summary.json, report.md, report.html there;
        - .json path: write JSON to that path, Markdown/HTML beside it with the same stem;
        - .md/.html path: write that artifact to the requested path, JSON as summary.json,
          and the remaining rich artifact beside it with the requested stem.
        """
        paths = self._resolve_output_paths(output_path)
        manifest = self.build_manifest(state, paths, workdir=workdir, state_path=state_path)
        payload = self.build_summary(state, audit_artifacts=manifest["audit_artifacts"])
        payload["manifest"] = manifest

        json_payload = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        markdown_payload = self.render_markdown(payload)
        html_payload = self.render_html(payload)

        atomic_write_text(paths["json"], json_payload + "\n")
        atomic_write_text(paths["manifest"], manifest_payload + "\n")
        atomic_write_text(paths["markdown"], markdown_payload)
        atomic_write_text(paths["html"], html_payload)
        return paths

    def generate_html(self, state: ProjectState, output_path: Union[str, Path]) -> None:
        """
        Backward-compatible HTML entrypoint.

        Unlike the old scaffold, this also writes adjacent JSON and Markdown
        artifacts because reports are now an artifact set.
        """
        self.generate_report(state, output_path)

    def build_summary(self, state: ProjectState, *, audit_artifacts: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        """Build a deterministic, strict-JSON-safe report payload."""
        status_counts = self._status_counts(state.sample_status)
        label_distribution = self._label_distribution(list(state.sample_labels.values()))
        review = self._review_metadata_summary(state.sample_review_metadata)
        metrics_history = [self._metric_record(index, record) for index, record in enumerate(state.metrics_history, start=1)]
        audit_artifacts_payload = _sanitize(audit_artifacts if audit_artifacts is not None else state.audit_artifacts)
        if not isinstance(audit_artifacts_payload, dict):
            audit_artifacts_payload = {}
        selection_refs_by_round = self._selection_audit_refs_by_round(
            audit_artifacts_payload.get("selection_audits", [])
        )
        rounds = [
            self._round_record(
                index,
                round_state,
                selection_audit=selection_refs_by_round.get(str(getattr(round_state, "round_id", ""))),
            )
            for index, round_state in enumerate(state.rounds, start=1)
        ]
        scheduler_snapshots = [
            {
                "round_id": round_record["round_id"],
                "status": round_record["status"],
                "snapshot": round_record["scheduler_snapshot"],
            }
            for round_record in rounds
            if round_record["scheduler_snapshot"]
        ]
        annotation_timeouts = [
            {
                "round_id": round_record["round_id"],
                "trace": round_record["annotation_timeout"],
            }
            for round_record in rounds
            if round_record["annotation_timeout"] is not None
        ]
        stop_trace = _sanitize(state.scheduler_state.get("stop_trace"))

        dataset_ref = state.dataset_ref
        dataset = None
        if dataset_ref is not None:
            dataset = {
                "source_type": _sanitize(dataset_ref.source_type),
                "source_path": _sanitize(dataset_ref.source_path),
                "schema": _sanitize(dataset_ref.schema),
                "fingerprint": _sanitize(dataset_ref.fingerprint),
                "fingerprint_config": _sanitize(dataset_ref.fingerprint_config),
            }

        return {
            "metadata": {
                "project_name": _sanitize(state.project_name),
                "state_version": _sanitize(state.state_version),
                "created_at": _sanitize(state.created_at),
                "created_at_iso": _timestamp_iso(state.created_at),
                "updated_at": _sanitize(state.updated_at),
                "updated_at_iso": _timestamp_iso(state.updated_at),
            },
            "dataset": dataset,
            "configuration": {
                "label_schema": _sanitize(state.label_schema),
                "annotation_policy": _sanitize(state.annotation_policy),
                "scheduler_config": _sanitize(state.scheduler_config),
                "label_backend_config": _sanitize(state.label_backend_config),
                "cache_config": _sanitize(state.cache_config),
                "split_config": _sanitize(state.split_config),
                "prelabel_config": _sanitize(state.prelabel_config),
                "splits": {name: len(sample_ids) for name, sample_ids in sorted(state.splits.items())},
            },
            "counts": {
                "total_samples": len(state.sample_status),
                "by_status": status_counts,
                "labeled": status_counts.get(SampleStatus.LABELED.value, 0),
                "unlabeled": status_counts.get(SampleStatus.UNLABELED.value, 0),
                "needs_review": status_counts.get(SampleStatus.NEEDS_REVIEW.value, 0),
                "invalid": status_counts.get(SampleStatus.INVALID.value, 0),
            },
            "labels": {
                "total_labeled": len(state.sample_labels),
                "distribution": label_distribution,
            },
            "review": review,
            "metrics_history": metrics_history,
            "rounds": rounds,
            "scheduler": {
                "state": _sanitize(state.scheduler_state),
                "snapshots": scheduler_snapshots,
                "stop_trace": stop_trace,
            },
            "traces": {
                "stop_trace": stop_trace,
                "annotation_timeouts": annotation_timeouts,
            },
            "audit": {
                "artifacts": audit_artifacts_payload,
                "event_log": _sanitize(audit_artifacts_payload.get("event_log")),
                "selection_audits": _sanitize(audit_artifacts_payload.get("selection_audits", [])),
            },
        }

    def build_manifest(
        self,
        state: ProjectState,
        paths: Mapping[str, Path] | None = None,
        *,
        workdir: Union[str, Path] | None = None,
        state_path: Union[str, Path] | None = None,
    ) -> Dict[str, Any]:
        """Build a strict-JSON reproducibility manifest for a report artifact set."""
        dataset_ref = state.dataset_ref
        dataset = None
        if dataset_ref is not None:
            dataset = {
                "source_type": dataset_ref.source_type,
                "source_path": dataset_ref.source_path,
                "schema": dataset_ref.schema,
                "fingerprint": dataset_ref.fingerprint,
                "fingerprint_config": dataset_ref.fingerprint_config,
            }

        generated_artifacts: Dict[str, str] = {}
        if paths is not None:
            generated_artifacts = {kind: path.name for kind, path in sorted(paths.items())}

        audit_artifacts = self._audit_manifest_artifacts(state, workdir=workdir, state_path=state_path)

        manifest = {
            "artifact_schema_version": REPORT_ARTIFACT_SCHEMA_VERSION,
            "generated_at_iso": datetime.now(tz=timezone.utc).isoformat(),
            "sdk": {
                "package_name": _PACKAGE_NAME,
                "package_version": _package_version(),
            },
            "runtime": {
                "python_version": sys.version,
                "python_version_info": {
                    "major": sys.version_info.major,
                    "minor": sys.version_info.minor,
                    "micro": sys.version_info.micro,
                },
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
            },
            "project": {
                "project_name": state.project_name,
                "state_version": state.state_version,
                "created_at": state.created_at,
                "created_at_iso": _timestamp_iso(state.created_at),
                "updated_at": state.updated_at,
                "updated_at_iso": _timestamp_iso(state.updated_at),
            },
            "dataset": dataset,
            "configuration": {
                "label_schema": state.label_schema,
                "annotation_policy": state.annotation_policy,
                "scheduler_config": state.scheduler_config,
                "label_backend_config": state.label_backend_config,
                "cache_config": state.cache_config,
                "split_config": state.split_config,
                "prelabel_config": state.prelabel_config,
                "splits": {name: len(sample_ids) for name, sample_ids in sorted(state.splits.items())},
            },
            "counts": {
                "samples": len(state.sample_status),
                "labels": len(state.sample_labels),
                "rounds": len(state.rounds),
                "metrics": len(state.metrics_history),
                "review_metadata": len(state.sample_review_metadata),
            },
            "artifacts": generated_artifacts,
            "audit_artifacts": audit_artifacts,
        }
        return _sanitize(manifest)

    def render_markdown(self, payload: Mapping[str, Any]) -> str:
        metadata = payload["metadata"]
        dataset = payload.get("dataset") or {}
        manifest = payload.get("manifest") or {}
        artifacts = manifest.get("artifacts") or {}
        review = payload["review"]
        counts = payload["counts"]
        labels = payload["labels"]
        review = payload["review"]
        lines = [
            f"# Active Learning Report: {_md(metadata['project_name'])}",
            "",
            "## Project Metadata",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Project | {_md(metadata['project_name'])} |",
            f"| State version | {_md(metadata['state_version'])} |",
            f"| Created | {_md(metadata['created_at_iso'])} |",
            f"| Updated | {_md(metadata['updated_at_iso'])} |",
            f"| Report artifact schema | {_md(manifest.get('artifact_schema_version'))} |",
            f"| Reproducibility manifest | {_md(artifacts.get('manifest'))} |",
            "",
            "## Dataset",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Source type | {_md(dataset.get('source_type'))} |",
            f"| Source path | {_md(dataset.get('source_path'))} |",
            f"| Fingerprint | {_md(dataset.get('fingerprint'))} |",
            f"| Schema | {_md(dataset.get('schema'))} |",
            "",
            "## Sample Counts",
            "",
            "| Status | Count |",
            "| --- | ---: |",
        ]
        for status, count in counts["by_status"].items():
            lines.append(f"| {_md(status)} | {count} |")

        lines.extend(
            [
                "",
                "## Label Distribution",
                "",
                "| Label | Count |",
                "| --- | ---: |",
            ]
        )
        for label, count in labels["distribution"].items():
            lines.append(f"| {_md(label)} | {count} |")

        lines.extend(
            [
                "",
                "## Annotation Review Metadata",
                "",
                f"Samples with review metadata: {review['total']}",
                "",
                "### Review Reasons",
                "",
                "| Reason | Count |",
                "| --- | ---: |",
            ]
        )
        for reason, count in review["by_reason"].items():
            lines.append(f"| {_md(reason)} | {count} |")

        lines.extend(
            [
                "",
                "## Metrics History",
                "",
                "| # | Step | Created | Metrics |",
                "| ---: | --- | --- | --- |",
            ]
        )
        for record in payload["metrics_history"]:
            lines.append(
                f"| {record['index']} | {_md(record['step'])} | {_md(record['created_at_iso'])} | "
                f"{_md(record['metrics'])} |"
            )

        lines.extend(
            [
                "",
                "## Round Timeline",
                "",
                "| # | Round | Status | Selected | Tasks | Resolved | Reward | Metrics after |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for round_record in payload["rounds"]:
            lines.append(
                f"| {round_record['index']} | {_md(round_record['round_id'])} | {_md(round_record['status'])} | "
                f"{round_record['selected_count']} | {round_record['task_count']} | {round_record['resolved_count']} | "
                f"{_md(round_record['reward'])} | {_md(round_record['metrics_after'])} |"
            )

        lines.extend(
            [
                "",
                "## Scheduler And Acquisition Snapshots",
                "",
                "```json",
                _md_json(payload["scheduler"]["snapshots"]),
                "```",
                "",
                "## Stop Trace",
                "",
                "```json",
                _md_json(payload["traces"]["stop_trace"]),
                "```",
                "",
            "## Annotation Timeout Traces",
            "",
            "```json",
            _md_json(payload["traces"]["annotation_timeouts"]),
            "```",
            "",
            "## Audit Artifacts",
            "",
            "```json",
            _md_json(payload.get("audit", {})),
            "```",
            "",
            "## Reproducibility Manifest",
                "",
                f"Standalone manifest artifact: `{_md(artifacts.get('manifest'))}`",
                "",
                "Generated artifacts:",
                "",
            ]
        )
        for kind, filename in artifacts.items():
            lines.append(f"- {_md(kind)}: `{_md(filename)}`")

        lines.extend(
            [
                "",
            ]
        )
        return "\n".join(lines)

    def render_html(self, payload: Mapping[str, Any]) -> str:
        metadata = payload["metadata"]
        dataset = payload.get("dataset") or {}
        manifest = payload.get("manifest") or {}
        artifacts = manifest.get("artifacts") or {}
        review = payload["review"]
        body = [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Active Learning Report</title>",
            "<style>",
            "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.5;margin:2rem;color:#17202a;background:#fbfcfd}",
            "h1,h2{color:#0f2d3f}table{border-collapse:collapse;width:100%;margin:1rem 0;background:white}",
            "th,td{border:1px solid #d7dee8;padding:.45rem .6rem;text-align:left;vertical-align:top}th{background:#edf3f8}",
            "pre{background:#10212f;color:#edf7ff;padding:1rem;overflow:auto;border-radius:.35rem}",
            ".muted{color:#52616f}",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Active Learning Report: {_h(metadata['project_name'])}</h1>",
            '<p class="muted">Generated from persisted project state.</p>',
            "<h2>Project Metadata</h2>",
            _html_table(
                ["Field", "Value"],
                [
                    ["Project", metadata["project_name"]],
                    ["State version", metadata["state_version"]],
                    ["Created", metadata["created_at_iso"]],
                    ["Updated", metadata["updated_at_iso"]],
                    ["Report artifact schema", manifest.get("artifact_schema_version")],
                    ["Reproducibility manifest", artifacts.get("manifest")],
                ],
            ),
            "<h2>Dataset</h2>",
            _html_table(
                ["Field", "Value"],
                [
                    ["Source type", dataset.get("source_type")],
                    ["Source path", dataset.get("source_path")],
                    ["Fingerprint", dataset.get("fingerprint")],
                    ["Schema", dataset.get("schema")],
                ],
            ),
            "<h2>Sample Counts</h2>",
            _html_table(["Status", "Count"], payload["counts"]["by_status"].items()),
            "<h2>Label Distribution</h2>",
            _html_table(["Label", "Count"], payload["labels"]["distribution"].items()),
            "<h2>Annotation Review Metadata</h2>",
            _html_table(
                ["Field", "Value"],
                [["Samples with review metadata", review["total"]], ["Review reasons", review["by_reason"]]],
            ),
            "<h2>Metrics History</h2>",
            _html_table(
                ["#", "Step", "Created", "Metrics"],
                [
                    [record["index"], record["step"], record["created_at_iso"], record["metrics"]]
                    for record in payload["metrics_history"]
                ],
            ),
            "<h2>Round Timeline</h2>",
            _html_table(
                ["#", "Round", "Status", "Selected", "Tasks", "Resolved", "Reward", "Metrics after"],
                [
                    [
                        round_record["index"],
                        round_record["round_id"],
                        round_record["status"],
                        round_record["selected_count"],
                        round_record["task_count"],
                        round_record["resolved_count"],
                        round_record["reward"],
                        round_record["metrics_after"],
                    ]
                    for round_record in payload["rounds"]
                ],
            ),
            "<h2>Scheduler And Acquisition Snapshots</h2>",
            _html_json(payload["scheduler"]["snapshots"]),
            "<h2>Stop Trace</h2>",
            _html_json(payload["traces"]["stop_trace"]),
            "<h2>Annotation Timeout Traces</h2>",
            _html_json(payload["traces"]["annotation_timeouts"]),
            "<h2>Audit Artifacts</h2>",
            _html_json(payload.get("audit", {})),
            "<h2>Reproducibility Manifest</h2>",
            _html_table(["Artifact", "Filename"], artifacts.items()),
            "</body>",
            "</html>",
            "",
        ]
        return "\n".join(body)

    def _resolve_output_paths(self, output_path: Union[str, Path]) -> Dict[str, Path]:
        path = Path(output_path)
        if path.suffix.lower() in _KNOWN_REPORT_FILENAMES:
            base_dir = path.parent if str(path.parent) else Path(".")
            stem = path.stem
            paths = {
                "json": base_dir / "summary.json",
                "markdown": base_dir / f"{stem}.md",
                "html": base_dir / f"{stem}.html",
            }
            if path.suffix.lower() == ".json":
                paths["json"] = path
            elif path.suffix.lower() == ".md":
                paths["markdown"] = path
            elif path.suffix.lower() == ".html":
                paths["html"] = path
            paths["manifest"] = self._manifest_path(base_dir, list(paths.values()))
            return paths

        return {
            "json": path / "summary.json",
            "markdown": path / "report.md",
            "html": path / "report.html",
            "manifest": path / "manifest.json",
        }

    def _manifest_path(self, base_dir: Path, reserved_paths: Sequence[Path]) -> Path:
        manifest_path = base_dir / "manifest.json"
        if manifest_path not in set(reserved_paths):
            return manifest_path
        return base_dir / "active_learning_manifest.json"

    def _audit_manifest_artifacts(
        self,
        state: ProjectState,
        *,
        workdir: Union[str, Path] | None,
        state_path: Union[str, Path] | None,
    ) -> Dict[str, Any]:
        root = Path(workdir) if workdir is not None else None
        artifacts: Dict[str, Any] = {
            "state": self._artifact_reference(Path(state_path), root=root) if state_path is not None else {"available": False},
            "event_log": _sanitize(state.audit_artifacts.get("event_log") or {"available": False}),
            "selection_audits": self._selection_audit_artifact_references(
                state.audit_artifacts.get("selection_audits", []),
                root=root,
            ),
        }
        event_log_ref = artifacts["event_log"]
        if isinstance(event_log_ref, dict) and root is not None and event_log_ref.get("path") and not event_log_ref.get("missing"):
            event_path = root / str(event_log_ref["path"])
            artifacts["event_log"] = self._artifact_reference(event_path, root=root)
        return artifacts

    def _selection_audit_artifact_references(self, refs: Any, *, root: Path | None) -> list[Any]:
        sanitized = _sanitize(refs)
        if not isinstance(sanitized, list):
            return []
        if root is None:
            return sanitized

        refreshed_refs: list[Any] = []
        for ref in sanitized:
            if not isinstance(ref, dict):
                refreshed_refs.append(ref)
                continue
            path_value = ref.get("path")
            if not path_value:
                refreshed_refs.append(ref)
                continue

            artifact_path = Path(str(path_value))
            if not artifact_path.is_absolute():
                artifact_path = root / artifact_path

            refreshed = self._artifact_reference(artifact_path, root=root)
            preserved_metadata = {
                key: value
                for key, value in ref.items()
                if key not in {"path", "sha256", "missing"}
            }
            preserved_metadata.update(refreshed)
            refreshed_refs.append(preserved_metadata)
        return refreshed_refs

    def _artifact_reference(self, path: Path, *, root: Path | None) -> Dict[str, Any]:
        try:
            rendered_path = path.relative_to(root).as_posix() if root is not None else str(path)
        except ValueError:
            rendered_path = str(path)
        reference: Dict[str, Any] = {"path": rendered_path}
        if path.exists():
            reference["sha256"] = sha256_file(path)
        else:
            reference["sha256"] = None
            reference["missing"] = True
        return reference

    def _status_counts(self, sample_status: Mapping[str, str]) -> Dict[str, int]:
        counts = Counter(str(status) for status in sample_status.values())
        ordered: Dict[str, int] = {}
        for enum_status in SampleStatus:
            ordered[enum_status.value] = counts.pop(enum_status.value, 0)
        for status_key in sorted(counts):
            ordered[status_key] = counts[status_key]
        return ordered

    def _label_distribution(self, labels: Iterable[Any]) -> Dict[str, int]:
        counts = Counter(_label_key(label) for label in labels)
        return {label: counts[label] for label in sorted(counts)}

    def _review_metadata_summary(self, sample_review_metadata: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        sanitized_samples = _sanitize(sample_review_metadata)
        if not isinstance(sanitized_samples, dict):
            sanitized_samples = {}
        reason_counts = Counter(
            str(metadata.get("reason", "unknown"))
            for metadata in sanitized_samples.values()
            if isinstance(metadata, Mapping)
        )
        return {
            "total": len(sanitized_samples),
            "by_reason": {reason: reason_counts[reason] for reason in sorted(reason_counts)},
            "samples": sanitized_samples,
        }

    def _metric_record(self, index: int, record: Any) -> Dict[str, Any]:
        return {
            "index": index,
            "step": _sanitize(getattr(record, "step", None)),
            "created_at": _sanitize(getattr(record, "created_at", None)),
            "created_at_iso": _timestamp_iso(getattr(record, "created_at", None)),
            "metrics": _sanitize(getattr(record, "metrics", {})),
        }

    def _selection_audit_refs_by_round(self, refs: Any) -> Dict[str, Mapping[str, Any]]:
        if not isinstance(refs, list):
            return {}
        by_round: Dict[str, Mapping[str, Any]] = {}
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            round_id = ref.get("round_id")
            if round_id is not None:
                by_round[str(round_id)] = ref
        return by_round

    def _round_record(
        self,
        index: int,
        round_state: Any,
        *,
        selection_audit: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        snapshot = _sanitize(getattr(round_state, "scheduler_snapshot", {}))
        return {
            "index": index,
            "round_id": _sanitize(getattr(round_state, "round_id", None)),
            "status": _sanitize(getattr(round_state, "status", None)),
            "created_at": _sanitize(getattr(round_state, "created_at", None)),
            "created_at_iso": _timestamp_iso(getattr(round_state, "created_at", None)),
            "updated_at": _sanitize(getattr(round_state, "updated_at", None)),
            "updated_at_iso": _timestamp_iso(getattr(round_state, "updated_at", None)),
            "selected_sample_ids": _sanitize(getattr(round_state, "selected_sample_ids", [])),
            "selected_count": len(getattr(round_state, "selected_sample_ids", []) or []),
            "task_ids": _sanitize(getattr(round_state, "task_ids", {})),
            "task_count": len(getattr(round_state, "task_ids", {}) or {}),
            "resolved": _sanitize(getattr(round_state, "resolved", {})),
            "resolved_count": len(getattr(round_state, "resolved", {}) or {}),
            "metrics_before": _sanitize(getattr(round_state, "metrics_before", {})),
            "metrics_after": _sanitize(getattr(round_state, "metrics_after", {})),
            "reward": _sanitize(getattr(round_state, "reward", None)),
            "scheduler_snapshot": snapshot,
            "selection_audit": _sanitize(
                selection_audit if selection_audit is not None else getattr(round_state, "selection_audit", {})
            ),
            "annotation_timeout": snapshot.get("annotation_timeout") if isinstance(snapshot, dict) else None,
            "error": _sanitize(getattr(round_state, "error", None)),
        }


def _sanitize(value: Any) -> Any:
    """Convert arbitrary report data into strict JSON-compatible values."""
    if not isinstance(value, type) and dataclasses.is_dataclass(value):
        return _sanitize(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return _sanitize(value.value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize(value[key]) for key in sorted(value.keys(), key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        sanitized_items = [_sanitize(item) for item in value]
        return sorted(sanitized_items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _sanitize(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _sanitize(item())
        except Exception:
            pass
    return repr(value)


def _timestamp_iso(value: Any) -> str | None:
    sanitized = _sanitize(value)
    if not isinstance(sanitized, numbers.Real):
        return None
    return datetime.fromtimestamp(float(sanitized), tz=timezone.utc).isoformat()


def _label_key(label: Any) -> str:
    sanitized = _sanitize(label)
    if isinstance(sanitized, str):
        return sanitized
    return json.dumps(sanitized, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _md(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    else:
        text = str(value)
    return _escape_markdown_plain_text(html.escape(_normalize_markdown_line_separators(text), quote=True))


def _md_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return _escape_markdown_plain_text(html.escape(_normalize_markdown_line_separators(text), quote=True))


def _normalize_markdown_line_separators(text: str) -> str:
    return text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\u2028", " ").replace("\u2029", " ")


def _escape_markdown_plain_text(text: str) -> str:
    escaped = text.replace("\\", "\\\\")
    for char in ("`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", ".", "!", "|", ">"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


def _h(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return html.escape(str(value), quote=True)


def _html_table(headers: Sequence[str], rows: Any) -> str:
    row_list = list(rows)
    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{_h(header)}</th>" for header in headers)
    parts.append("</tr></thead><tbody>")
    for row in row_list:
        cells = list(row) if isinstance(row, (list, tuple)) else list(row)
        parts.append("<tr>")
        parts.extend(f"<td>{_h(cell)}</td>" for cell in cells)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _html_json(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return f"<pre>{html.escape(rendered, quote=False)}</pre>"


def _package_version() -> str | None:
    try:
        return importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None
