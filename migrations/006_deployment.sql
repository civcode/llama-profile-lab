CREATE TABLE deployment_candidate (
    id TEXT PRIMARY KEY,
    deployment_hash TEXT NOT NULL UNIQUE,
    workload_suite_id TEXT NOT NULL REFERENCES workload_suite(id),
    resource_policy_json TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE deployment_instance (
    deployment_candidate_id TEXT NOT NULL
        REFERENCES deployment_candidate(id) ON DELETE CASCADE,
    instance_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    role TEXT NOT NULL,
    model_artifact_id TEXT NOT NULL,
    binary_id TEXT NOT NULL REFERENCES binary(id),
    requested_placement_json TEXT NOT NULL,
    server_identity TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    PRIMARY KEY (deployment_candidate_id, instance_id),
    UNIQUE (deployment_candidate_id, ordinal)
);

CREATE TABLE deployment_placement (
    id TEXT PRIMARY KEY,
    placement_hash TEXT NOT NULL UNIQUE,
    deployment_candidate_id TEXT NOT NULL
        REFERENCES deployment_candidate(id),
    host_id TEXT NOT NULL REFERENCES host(id),
    feasibility TEXT NOT NULL,
    request_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (feasibility IN ('pending', 'feasible', 'infeasible', 'estimate_failed')),
    UNIQUE (id, deployment_candidate_id)
);

CREATE TABLE deployment_instance_placement (
    deployment_placement_id TEXT NOT NULL,
    deployment_candidate_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    resolved_placement_id TEXT NOT NULL REFERENCES resolved_placement(id),
    PRIMARY KEY (deployment_placement_id, instance_id),
    FOREIGN KEY (deployment_placement_id, deployment_candidate_id)
        REFERENCES deployment_placement(id, deployment_candidate_id)
        ON DELETE CASCADE,
    FOREIGN KEY (deployment_candidate_id, instance_id)
        REFERENCES deployment_instance(deployment_candidate_id, instance_id)
);

CREATE TABLE placement_device_memory (
    id TEXT PRIMARY KEY,
    deployment_placement_id TEXT NOT NULL,
    deployment_candidate_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    model_bytes INTEGER NOT NULL CHECK (model_bytes >= 0),
    context_bytes INTEGER NOT NULL CHECK (context_bytes >= 0),
    compute_bytes INTEGER NOT NULL CHECK (compute_bytes >= 0),
    total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0),
    device_total_bytes INTEGER NOT NULL CHECK (device_total_bytes >= 0),
    device_free_bytes INTEGER NOT NULL CHECK (device_free_bytes >= 0),
    source TEXT NOT NULL,
    measured_at TEXT,
    CHECK (total_bytes = model_bytes + context_bytes + compute_bytes),
    CHECK (device_free_bytes <= device_total_bytes),
    UNIQUE (deployment_placement_id, instance_id, device_id),
    FOREIGN KEY (deployment_placement_id, deployment_candidate_id)
        REFERENCES deployment_placement(id, deployment_candidate_id)
        ON DELETE CASCADE,
    FOREIGN KEY (deployment_candidate_id, instance_id)
        REFERENCES deployment_instance(deployment_candidate_id, instance_id)
);

CREATE TABLE deployment_device_allocation (
    deployment_placement_id TEXT NOT NULL REFERENCES deployment_placement(id)
        ON DELETE CASCADE,
    device_id TEXT NOT NULL,
    projected_bytes INTEGER NOT NULL CHECK (projected_bytes >= 0),
    reserved_margin_bytes INTEGER NOT NULL CHECK (reserved_margin_bytes >= 0),
    device_total_bytes INTEGER NOT NULL CHECK (device_total_bytes >= 0),
    projected_free_bytes INTEGER NOT NULL CHECK (projected_free_bytes >= 0),
    CHECK (
        projected_bytes + reserved_margin_bytes + projected_free_bytes
        = device_total_bytes
    ),
    PRIMARY KEY (deployment_placement_id, device_id)
);

CREATE TABLE deployment_run (
    id TEXT PRIMARY KEY,
    deployment_candidate_id TEXT NOT NULL
        REFERENCES deployment_candidate(id),
    deployment_placement_id TEXT REFERENCES deployment_placement(id),
    workload_case_id TEXT REFERENCES workload_case(id),
    status TEXT NOT NULL,
    quality TEXT,
    quality_details_json TEXT,
    failure_kind TEXT,
    failure_details_json TEXT,
    started_at TEXT,
    finished_at TEXT,
    duration_ns INTEGER CHECK (duration_ns IS NULL OR duration_ns >= 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (status IN (
        'planned', 'starting', 'ready', 'running',
        'completed', 'cancelled', 'failed'
    )),
    CHECK (
        failure_kind IS NULL OR failure_kind IN (
            'device_capability_mismatch',
            'memory_projection_failed',
            'memory_infeasible',
            'server_start_failed',
            'server_oom',
            'concurrent_workload_failed',
            'member_timeout',
            'member_crash',
            'telemetry_incomplete',
            'runtime_memory_margin_violated',
            'output_validation_failed'
        )
    ),
    UNIQUE (id, deployment_candidate_id)
);

CREATE TABLE deployment_run_member (
    deployment_run_id TEXT NOT NULL,
    deployment_candidate_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    server_run_id TEXT REFERENCES server_run(id),
    client_run_id TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (deployment_run_id, instance_id),
    FOREIGN KEY (deployment_run_id, deployment_candidate_id)
        REFERENCES deployment_run(id, deployment_candidate_id)
        ON DELETE CASCADE,
    FOREIGN KEY (deployment_candidate_id, instance_id)
        REFERENCES deployment_instance(deployment_candidate_id, instance_id)
);

CREATE INDEX idx_deployment_instance_candidate
ON deployment_instance(candidate_id);

CREATE INDEX idx_deployment_instance_binary
ON deployment_instance(binary_id);

CREATE INDEX idx_deployment_placement_candidate
ON deployment_placement(deployment_candidate_id, created_at);

CREATE INDEX idx_placement_device_memory_device
ON placement_device_memory(device_id, deployment_placement_id);

CREATE INDEX idx_deployment_run_candidate
ON deployment_run(deployment_candidate_id, created_at);

CREATE INDEX idx_deployment_run_placement
ON deployment_run(deployment_placement_id, created_at);
