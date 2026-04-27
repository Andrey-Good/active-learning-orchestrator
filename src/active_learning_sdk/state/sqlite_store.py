from __future__ import annotations

"""
SQLite-backed project state store scaffold.

This store is intentionally conservative: it persists the complete `ProjectState`
JSON snapshot in `config['project_state']` so it can satisfy the existing
`StateStore` contract, and also mirrors the most important entities into
normalized tables for fast status queries, migrations, analytics, and future
incremental writes.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, List, Optional, Union

from ..exceptions import StateCorruptedError
from ..types import SampleStatus
from .abc_store import ABCStore, StatusLike, normalize_status
from .store import ProjectState, state_from_json_dict, state_to_json_dict


class SqliteStateStore(ABCStore):
    """SQLite implementation of the project state store.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. A typical value is
        `workdir / "state.sqlite3"`.
    migrations_dir:
        Optional custom migrations directory. Defaults to
        `state/migrations` next to this file.
    """

    PROJECT_STATE_KEY = "project_state"

    def __init__(self, db_path: Union[str, Path], migrations_dir: Optional[Union[str, Path]] = None) -> None:
        self.db_path = Path(db_path)
        self.migrations_dir = Path(migrations_dir) if migrations_dir is not None else Path(__file__).parent / "migrations"
        self._conn: Optional[sqlite3.Connection] = None
        self._in_transaction = False

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        return self._conn

    def initialize(self) -> None:
        """Create database tables and apply the initial migration."""
        migration_path = self.migrations_dir / "001_initial.sql"
        if not migration_path.exists():
            raise StateCorruptedError(f"Missing SQLite migration: {migration_path}")
        self.conn.executescript(migration_path.read_text(encoding="utf-8"))

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._in_transaction = False

    def load(self) -> ProjectState:
        """Load the complete `ProjectState` JSON snapshot from SQLite."""
        self.initialize()
        row = self.conn.execute("SELECT value FROM config WHERE key = ?", (self.PROJECT_STATE_KEY,)).fetchone()
        if row is None:
            raise StateCorruptedError(f"No project state found in SQLite database: {self.db_path}")
        try:
            payload = json.loads(str(row["value"]))
            return state_from_json_dict(payload)
        except json.JSONDecodeError as error:
            raise StateCorruptedError(f"Invalid JSON project state in SQLite: {error}") from error
        except Exception as error:
            raise StateCorruptedError(f"Failed to load SQLite project state: {error}") from error

    def save_atomic(self, state: ProjectState) -> None:
        """Persist the full state and normalized mirrors in one SQLite transaction."""
        self.initialize()
        started_here = False
        if not self._in_transaction:
            self.begin_transaction()
            started_here = True
        try:
            payload = json.dumps(state_to_json_dict(state), ensure_ascii=False, separators=(",", ":"))
            self.conn.execute(
                """
                INSERT INTO config(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (self.PROJECT_STATE_KEY, payload),
            )
            self._sync_normalized_tables(state)
            if started_here:
                self.commit()
        except Exception:
            if started_here:
                self.rollback()
            raise

    def begin_transaction(self) -> None:
        if self._in_transaction:
            return
        self.conn.execute("BEGIN IMMEDIATE")
        self._in_transaction = True

    def commit(self) -> None:
        if not self._in_transaction:
            return
        self.conn.execute("COMMIT")
        self._in_transaction = False

    def rollback(self) -> None:
        if not self._in_transaction:
            return
        self.conn.execute("ROLLBACK")
        self._in_transaction = False

    def get_sample_status(self, sample_id: str) -> Optional[str]:
        self.initialize()
        row = self.conn.execute("SELECT status FROM samples WHERE id = ?", (sample_id,)).fetchone()
        return None if row is None else str(row["status"])

    def set_sample_status(self, sample_id: str, status: StatusLike) -> None:
        """Set one sample status and keep the full JSON snapshot consistent when present."""
        wanted = normalize_status(status)
        try:
            state = self.load()
        except StateCorruptedError:
            self.initialize()
            now = time.time()
            self.conn.execute(
                """
                INSERT INTO samples(id, status, created_at, updated_at) VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at
                """,
                (sample_id, wanted, now, now),
            )
            return
        state.sample_status[sample_id] = wanted
        self.save_atomic(state)

    def get_samples_by_status(self, status: StatusLike) -> List[str]:
        self.initialize()
        wanted = normalize_status(status)
        rows = self.conn.execute("SELECT id FROM samples WHERE status = ? ORDER BY id", (wanted,)).fetchall()
        return [str(row["id"]) for row in rows]

    def _sync_normalized_tables(self, state: ProjectState) -> None:
        now = time.time()

        for round_state in state.rounds:
            params = json.dumps(round_state.scheduler_snapshot, ensure_ascii=False, separators=(",", ":"))
            f1 = round_state.metrics_after.get("f1") or round_state.metrics_after.get("macro_f1")
            self.conn.execute(
                """
                INSERT INTO rounds(id, status, strategy, params, f1, sha256_weights, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    strategy = excluded.strategy,
                    params = excluded.params,
                    f1 = excluded.f1,
                    sha256_weights = excluded.sha256_weights,
                    updated_at = excluded.updated_at
                """,
                (
                    round_state.round_id,
                    round_state.status.value,
                    str(round_state.scheduler_snapshot.get("strategy", "")) or None,
                    params,
                    float(f1) if f1 is not None else None,
                    None,
                    round_state.created_at,
                    round_state.updated_at,
                ),
            )

        latest_round_by_sample: dict[str, str] = {}
        task_id_by_pair: dict[tuple[str, str], str] = {}
        for round_state in state.rounds:
            for sample_id in round_state.selected_sample_ids:
                latest_round_by_sample[sample_id] = round_state.round_id
                if sample_id in round_state.task_ids:
                    task_id_by_pair[(round_state.round_id, sample_id)] = round_state.task_ids[sample_id]

        for sample_id, status_value in state.sample_status.items():
            label = state.sample_labels.get(sample_id)
            label_payload = json.dumps(label, ensure_ascii=False, separators=(",", ":")) if label is not None else None
            self.conn.execute(
                """
                INSERT INTO samples(id, status, label, round_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    label = excluded.label,
                    round_id = excluded.round_id,
                    updated_at = excluded.updated_at
                """,
                (sample_id, status_value, label_payload, latest_round_by_sample.get(sample_id), now, now),
            )

        self.conn.execute("DELETE FROM round_samples")
        for round_state in state.rounds:
            for sample_id in round_state.selected_sample_ids:
                self.conn.execute(
                    """
                    INSERT INTO round_samples(round_id, sample_id, task_id) VALUES (?, ?, ?)
                    ON CONFLICT(round_id, sample_id) DO UPDATE SET task_id = excluded.task_id
                    """,
                    (round_state.round_id, sample_id, task_id_by_pair.get((round_state.round_id, sample_id))),
                )

        self.conn.execute("DELETE FROM metrics")
        for round_state in state.rounds:
            for step, metrics in (("before", round_state.metrics_before), ("after", round_state.metrics_after)):
                if metrics:
                    self.conn.execute(
                        "INSERT INTO metrics(round_id, step, metrics, created_at) VALUES (?, ?, ?, ?)",
                        (
                            round_state.round_id,
                            step,
                            json.dumps(metrics, ensure_ascii=False, separators=(",", ":")),
                            round_state.updated_at,
                        ),
                    )
        for metric_record in state.metrics_history:
            self.conn.execute(
                "INSERT INTO metrics(round_id, step, metrics, created_at) VALUES (?, ?, ?, ?)",
                (
                    None,
                    metric_record.step,
                    json.dumps(metric_record.metrics, ensure_ascii=False, separators=(",", ":")),
                    metric_record.created_at,
                ),
            )
