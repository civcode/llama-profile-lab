CREATE TABLE deployment_rejection (
    id TEXT PRIMARY KEY,
    deployment_candidate_id TEXT NOT NULL
        REFERENCES deployment_candidate(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    reason TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_deployment_rejection_candidate
ON deployment_rejection(deployment_candidate_id, created_at);
