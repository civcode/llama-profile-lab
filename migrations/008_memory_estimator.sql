CREATE TABLE accelerator_device (
    id TEXT PRIMARY KEY,
    host_id TEXT NOT NULL REFERENCES host(id),
    binary_id TEXT NOT NULL REFERENCES binary(id),
    logical_device_name TEXT NOT NULL,
    backend TEXT NOT NULL,
    mapping_status TEXT NOT NULL,
    physical_device_key TEXT,
    pci_bus_id TEXT,
    uuid TEXT,
    vendor TEXT,
    product_name TEXT,
    total_memory_bytes INTEGER,
    free_memory_bytes INTEGER,
    driver TEXT,
    runtime_metadata_json TEXT NOT NULL DEFAULT '{}',
    raw_output TEXT NOT NULL DEFAULT '',
    observed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (mapping_status IN ('mapped', 'unresolved')),
    CHECK (total_memory_bytes IS NULL OR total_memory_bytes >= 0),
    CHECK (free_memory_bytes IS NULL OR free_memory_bytes >= 0),
    CHECK (
        total_memory_bytes IS NULL
        OR free_memory_bytes IS NULL
        OR free_memory_bytes <= total_memory_bytes
    ),
    CHECK (
        mapping_status != 'mapped'
        OR physical_device_key IS NOT NULL
    ),
    UNIQUE (host_id, binary_id, logical_device_name)
);

CREATE TABLE memory_estimate_attempt (
    id TEXT PRIMARY KEY,
    cache_hash TEXT NOT NULL,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    host_id TEXT NOT NULL REFERENCES host(id),
    helper_binary_id TEXT NOT NULL REFERENCES binary(id),
    model_artifact_id TEXT NOT NULL,
    request_json TEXT NOT NULL,
    argv_json TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ns INTEGER CHECK (duration_ns IS NULL OR duration_ns >= 0),
    exit_code INTEGER,
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT '',
    failure_details_json TEXT,
    CHECK (status IN (
        'running', 'completed', 'failed', 'parser_failed',
        'timeout', 'interrupted', 'cancelled', 'binary_changed'
    ))
);

CREATE TABLE memory_estimate (
    id TEXT PRIMARY KEY,
    cache_hash TEXT NOT NULL UNIQUE,
    attempt_id TEXT NOT NULL REFERENCES memory_estimate_attempt(id),
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    host_id TEXT NOT NULL REFERENCES host(id),
    helper_binary_id TEXT NOT NULL REFERENCES binary(id),
    model_artifact_id TEXT NOT NULL,
    identity_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE memory_estimate_device (
    memory_estimate_id TEXT NOT NULL
        REFERENCES memory_estimate(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    logical_device_name TEXT NOT NULL,
    model_bytes INTEGER NOT NULL CHECK (model_bytes >= 0),
    context_bytes INTEGER NOT NULL CHECK (context_bytes >= 0),
    compute_bytes INTEGER NOT NULL CHECK (compute_bytes >= 0),
    total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0),
    device_total_bytes INTEGER NOT NULL CHECK (device_total_bytes >= 0),
    device_free_bytes INTEGER NOT NULL CHECK (device_free_bytes >= 0),
    CHECK (total_bytes = model_bytes + context_bytes + compute_bytes),
    CHECK (device_free_bytes <= device_total_bytes),
    PRIMARY KEY (memory_estimate_id, logical_device_name),
    UNIQUE (memory_estimate_id, ordinal)
);

CREATE INDEX idx_accelerator_device_host_binary
ON accelerator_device(host_id, binary_id);

CREATE INDEX idx_memory_estimate_attempt_cache
ON memory_estimate_attempt(cache_hash, started_at);

CREATE INDEX idx_memory_estimate_candidate
ON memory_estimate(candidate_id, created_at);

CREATE INDEX idx_memory_estimate_device_name
ON memory_estimate_device(logical_device_name, memory_estimate_id);
