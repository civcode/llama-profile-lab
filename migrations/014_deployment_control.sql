CREATE TABLE deployment_control_request (
    deployment_candidate_id TEXT PRIMARY KEY
        REFERENCES deployment_candidate(id) ON DELETE CASCADE,
    action TEXT NOT NULL CHECK (action IN ('pause', 'cancel')),
    requested_at TEXT NOT NULL
        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_deployment_control_action
ON deployment_control_request(action, requested_at);
