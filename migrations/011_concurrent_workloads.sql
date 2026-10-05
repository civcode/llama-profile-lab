CREATE TABLE deployment_concurrent_workload_case (
    id TEXT PRIMARY KEY,
    deployment_candidate_id TEXT NOT NULL
        REFERENCES deployment_candidate(id) ON DELETE CASCADE,
    case_hash TEXT NOT NULL,
    phase TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (phase IN ('dd', 'pp', 'pd', 'dp')),
    UNIQUE (deployment_candidate_id, case_hash)
);

CREATE TABLE deployment_workload_run (
    id TEXT PRIMARY KEY,
    deployment_run_id TEXT NOT NULL
        REFERENCES deployment_run(id) ON DELETE CASCADE,
    workload_case_id TEXT NOT NULL
        REFERENCES deployment_concurrent_workload_case(id),
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    quality TEXT,
    correctness_valid INTEGER NOT NULL DEFAULT 1,
    barrier_release_ns INTEGER,
    overlap_start_ns INTEGER,
    overlap_end_ns INTEGER,
    overlap_duration_ns INTEGER,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    decode_tokens INTEGER NOT NULL DEFAULT 0,
    combined_prompt_tps REAL,
    combined_decode_tps REAL,
    min_retention REAL,
    failure_kind TEXT,
    failure_details_json TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (phase IN ('dd', 'pp', 'pd', 'dp')),
    CHECK (status IN ('running', 'completed', 'failed', 'cancelled', 'invalid')),
    CHECK (correctness_valid IN (0, 1)),
    CHECK (barrier_release_ns IS NULL OR barrier_release_ns >= 0),
    CHECK (overlap_start_ns IS NULL OR overlap_start_ns >= 0),
    CHECK (overlap_end_ns IS NULL OR overlap_end_ns >= 0),
    CHECK (overlap_duration_ns IS NULL OR overlap_duration_ns > 0),
    CHECK (prompt_tokens >= 0),
    CHECK (decode_tokens >= 0),
    CHECK (combined_prompt_tps IS NULL OR combined_prompt_tps >= 0),
    CHECK (combined_decode_tps IS NULL OR combined_decode_tps >= 0),
    CHECK (min_retention IS NULL OR min_retention >= 0)
);

CREATE TABLE deployment_standalone_baseline (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    resolved_placement_id TEXT NOT NULL REFERENCES resolved_placement(id),
    host_id TEXT NOT NULL REFERENCES host(id),
    binary_id TEXT NOT NULL REFERENCES binary(id),
    mode TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    generate_tokens INTEGER NOT NULL,
    depth_tokens INTEGER NOT NULL,
    throughput_tps REAL NOT NULL,
    latency_ms REAL,
    source_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (mode IN ('prefill', 'decode')),
    CHECK (prompt_tokens >= 0),
    CHECK (generate_tokens >= 0),
    CHECK (depth_tokens >= 0),
    CHECK (throughput_tps > 0),
    CHECK (latency_ms IS NULL OR latency_ms >= 0)
);

CREATE TABLE deployment_workload_member (
    deployment_workload_run_id TEXT NOT NULL
        REFERENCES deployment_workload_run(id) ON DELETE CASCADE,
    instance_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    client_ready_ns INTEGER NOT NULL CHECK (client_ready_ns >= 0),
    barrier_release_ns INTEGER NOT NULL CHECK (barrier_release_ns >= 0),
    first_request_ns INTEGER,
    first_token_ns INTEGER,
    last_token_ns INTEGER,
    finished_ns INTEGER,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    decode_tokens INTEGER NOT NULL DEFAULT 0,
    native_prompt_tps REAL,
    native_decode_tps REAL,
    overlap_prompt_tokens INTEGER NOT NULL DEFAULT 0,
    overlap_decode_tokens INTEGER NOT NULL DEFAULT 0,
    overlap_prompt_tps REAL,
    overlap_decode_tps REAL,
    latency_ms REAL,
    standalone_baseline_id TEXT
        REFERENCES deployment_standalone_baseline(id),
    standalone_tps REAL,
    retention REAL,
    throughput_loss_pct REAL,
    correctness_valid INTEGER NOT NULL DEFAULT 1,
    raw_json TEXT NOT NULL DEFAULT '{}',
    failure_details_json TEXT,
    PRIMARY KEY (deployment_workload_run_id, instance_id),
    UNIQUE (deployment_workload_run_id, ordinal),
    CHECK (mode IN ('prefill', 'decode')),
    CHECK (status IN (
        'ready', 'completed', 'failed', 'timeout', 'cancelled', 'invalid'
    )),
    CHECK (first_request_ns IS NULL OR first_request_ns >= 0),
    CHECK (first_token_ns IS NULL OR first_token_ns >= 0),
    CHECK (last_token_ns IS NULL OR last_token_ns >= 0),
    CHECK (finished_ns IS NULL OR finished_ns >= 0),
    CHECK (prompt_tokens >= 0),
    CHECK (decode_tokens >= 0),
    CHECK (native_prompt_tps IS NULL OR native_prompt_tps >= 0),
    CHECK (native_decode_tps IS NULL OR native_decode_tps >= 0),
    CHECK (overlap_prompt_tokens >= 0),
    CHECK (overlap_decode_tokens >= 0),
    CHECK (overlap_prompt_tps IS NULL OR overlap_prompt_tps >= 0),
    CHECK (overlap_decode_tps IS NULL OR overlap_decode_tps >= 0),
    CHECK (latency_ms IS NULL OR latency_ms >= 0),
    CHECK (standalone_tps IS NULL OR standalone_tps > 0),
    CHECK (retention IS NULL OR retention >= 0),
    CHECK (correctness_valid IN (0, 1))
);

CREATE INDEX idx_deployment_workload_case_candidate
ON deployment_concurrent_workload_case(deployment_candidate_id, phase);

CREATE INDEX idx_deployment_workload_run_parent
ON deployment_workload_run(deployment_run_id, phase, created_at);

CREATE INDEX idx_deployment_baseline_lookup
ON deployment_standalone_baseline(
    candidate_id, resolved_placement_id, host_id, binary_id,
    mode, prompt_tokens, generate_tokens, depth_tokens
);

CREATE INDEX idx_deployment_workload_member_instance
ON deployment_workload_member(instance_id, deployment_workload_run_id);
