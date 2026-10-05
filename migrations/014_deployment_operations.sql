CREATE TABLE deployment_operation (
    id TEXT PRIMARY KEY,
    base_deployment_candidate_id TEXT NOT NULL
        REFERENCES deployment_candidate(id),
    deployment_placement_id TEXT NOT NULL
        REFERENCES deployment_placement(id),
    deployment_run_id TEXT REFERENCES deployment_run(id),
    request_json TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_action TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (status IN (
        'running', 'pausing', 'cancelling',
        'completed', 'paused', 'cancelled', 'failed'
    )),
    CHECK (
        requested_action IS NULL
        OR requested_action IN ('pause', 'cancel')
    )
);

CREATE INDEX idx_deployment_operation_candidate
ON deployment_operation(base_deployment_candidate_id, created_at);

CREATE INDEX idx_deployment_operation_status
ON deployment_operation(status, created_at);
