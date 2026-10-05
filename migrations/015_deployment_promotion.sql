CREATE TABLE deployment_promotion_proposal (
    id TEXT PRIMARY KEY,
    base_deployment_candidate_id TEXT NOT NULL
        REFERENCES deployment_candidate(id),
    deployment_candidate_id TEXT NOT NULL,
    deployment_placement_id TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    changes_json TEXT NOT NULL,
    source_snapshot_json TEXT NOT NULL,
    proposed_snapshot_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    patch TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (deployment_placement_id, deployment_candidate_id)
        REFERENCES deployment_placement(id, deployment_candidate_id)
);

CREATE INDEX idx_deployment_promotion_base
ON deployment_promotion_proposal(base_deployment_candidate_id, created_at);

CREATE INDEX idx_deployment_promotion_placement
ON deployment_promotion_proposal(deployment_placement_id, created_at);
