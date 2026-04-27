-- Initial SQLite schema for local Active Learning SDK state.
-- Keep this migration idempotent: it may be executed on a fresh database during
-- store initialization.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS rounds (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    strategy TEXT,
    params TEXT,
    f1 REAL,
    sha256_weights TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    CHECK (params IS NULL OR json_valid(params))
);

CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'unlabeled',
    score REAL,
    label TEXT,
    round_id TEXT,
    created_at REAL NOT NULL DEFAULT (unixepoch()),
    updated_at REAL NOT NULL DEFAULT (unixepoch()),
    FOREIGN KEY (round_id) REFERENCES rounds(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS round_samples (
    round_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    task_id TEXT,
    PRIMARY KEY (round_id, sample_id),
    FOREIGN KEY (round_id) REFERENCES rounds(id) ON DELETE CASCADE,
    FOREIGN KEY (sample_id) REFERENCES samples(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id TEXT,
    step TEXT NOT NULL,
    metrics TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch()),
    FOREIGN KEY (round_id) REFERENCES rounds(id) ON DELETE CASCADE,
    CHECK (json_valid(metrics))
);

CREATE TABLE IF NOT EXISTS outbox_pending (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch()),
    CHECK (json_valid(payload))
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status);
CREATE INDEX IF NOT EXISTS idx_samples_round_id ON samples(round_id);
CREATE INDEX IF NOT EXISTS idx_round_samples_sample_id ON round_samples(sample_id);
CREATE INDEX IF NOT EXISTS idx_metrics_round_id ON metrics(round_id);
CREATE INDEX IF NOT EXISTS idx_outbox_pending_event_type ON outbox_pending(event_type);

INSERT OR IGNORE INTO schema_version(version) VALUES (1);
