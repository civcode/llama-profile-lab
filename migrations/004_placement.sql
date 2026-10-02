CREATE TABLE placement_attempt (
    id TEXT PRIMARY KEY,
    placement_hash TEXT NOT NULL,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    host_id TEXT NOT NULL REFERENCES host(id),
    binary_id TEXT NOT NULL REFERENCES binary(id),
    model_path TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ns INTEGER CHECK (duration_ns IS NULL OR duration_ns >= 0),
    status TEXT NOT NULL,
    exit_code INTEGER,
    argv_json TEXT NOT NULL,
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT '',
    raw_result_json TEXT,
    CHECK (status IN (
        'running', 'completed', 'fit_failed', 'timeout',
        'parser_failed', 'interrupted', 'cancelled'
    ))
);

ALTER TABLE resolved_placement
ADD COLUMN fit_attempt_id TEXT REFERENCES placement_attempt(id);

ALTER TABLE resolved_placement
ADD COLUMN devices_json TEXT NOT NULL DEFAULT '"auto"';

ALTER TABLE resolved_placement
ADD COLUMN request_json TEXT NOT NULL DEFAULT '{}';

CREATE INDEX idx_placement_attempt_hash
ON placement_attempt(placement_hash, started_at);

CREATE INDEX idx_placement_attempt_candidate
ON placement_attempt(candidate_id, started_at);
