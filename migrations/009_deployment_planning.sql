CREATE TABLE deployment_plan (
    id TEXT PRIMARY KEY,
    base_deployment_candidate_id TEXT NOT NULL
        REFERENCES deployment_candidate(id),
    host_id TEXT NOT NULL REFERENCES host(id),
    search_hash TEXT NOT NULL,
    search_json TEXT NOT NULL,
    request_json TEXT NOT NULL DEFAULT '{}',
    raw_combinations INTEGER NOT NULL CHECK (raw_combinations >= 0),
    rejected_by_constraints INTEGER NOT NULL CHECK (rejected_by_constraints >= 0),
    duplicate_candidates INTEGER NOT NULL CHECK (duplicate_candidates >= 0),
    symmetry_reduced INTEGER NOT NULL CHECK (symmetry_reduced >= 0),
    capability_rejected INTEGER NOT NULL CHECK (capability_rejected >= 0),
    estimate_failed INTEGER NOT NULL CHECK (estimate_failed >= 0),
    memory_rejected INTEGER NOT NULL CHECK (memory_rejected >= 0),
    valid_count INTEGER NOT NULL CHECK (valid_count >= 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE deployment_plan_case (
    deployment_plan_id TEXT NOT NULL REFERENCES deployment_plan(id)
        ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    case_hash TEXT NOT NULL,
    deployment_candidate_id TEXT NOT NULL
        REFERENCES deployment_candidate(id),
    deployment_placement_id TEXT NOT NULL
        REFERENCES deployment_placement(id),
    generation_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (deployment_plan_id, ordinal),
    UNIQUE (deployment_plan_id, case_hash)
);

CREATE INDEX idx_deployment_plan_base
ON deployment_plan(base_deployment_candidate_id, created_at);

CREATE INDEX idx_deployment_plan_case_candidate
ON deployment_plan_case(deployment_candidate_id, deployment_plan_id);
