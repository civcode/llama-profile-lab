CREATE TABLE host (
    id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL,
    hardware_fingerprint TEXT NOT NULL UNIQUE,
    cpu_json TEXT NOT NULL,
    ram_bytes INTEGER NOT NULL CHECK (ram_bytes >= 0),
    gpu_json TEXT NOT NULL,
    os_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE binary (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    mtime_ns INTEGER NOT NULL CHECK (mtime_ns >= 0),
    git_commit TEXT,
    git_branch TEXT,
    git_dirty INTEGER CHECK (git_dirty IN (0, 1)),
    build_number TEXT,
    build_info_json TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE model (
    id TEXT PRIMARY KEY,
    identity_hash TEXT NOT NULL UNIQUE,
    architecture TEXT,
    parameter_count INTEGER CHECK (parameter_count IS NULL OR parameter_count >= 0),
    quantization TEXT,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE model_file (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model(id) ON DELETE CASCADE,
    part_index INTEGER NOT NULL CHECK (part_index >= 0),
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    UNIQUE (model_id, part_index),
    UNIQUE (model_id, sha256)
);

CREATE TABLE candidate (
    id TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL UNIQUE,
    target_model_id TEXT NOT NULL,
    draft_model_id TEXT,
    context_size INTEGER NOT NULL CHECK (context_size > 0),
    batch_size INTEGER NOT NULL CHECK (batch_size > 0),
    ubatch_size INTEGER NOT NULL CHECK (ubatch_size > 0),
    cache_type_k TEXT NOT NULL,
    cache_type_v TEXT NOT NULL,
    flash_attn TEXT NOT NULL,
    fit_target_mib INTEGER CHECK (fit_target_mib IS NULL OR fit_target_mib >= 0),
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (ubatch_size <= batch_size)
);

CREATE TABLE search_space (
    id TEXT PRIMARY KEY,
    definition_hash TEXT NOT NULL UNIQUE,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE workload_suite (
    id TEXT PRIMARY KEY,
    definition_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE workload_case (
    id TEXT PRIMARY KEY,
    workload_hash TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    prompt_tokens INTEGER,
    generate_tokens INTEGER,
    depth_tokens INTEGER,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (prompt_tokens IS NULL OR prompt_tokens >= 0),
    CHECK (generate_tokens IS NULL OR generate_tokens >= 0),
    CHECK (depth_tokens IS NULL OR depth_tokens >= 0)
);

CREATE TABLE measurement_policy (
    id TEXT PRIMARY KEY,
    policy_hash TEXT NOT NULL UNIQUE,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE experiment (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    base_candidate_id TEXT NOT NULL REFERENCES candidate(id),
    search_space_id TEXT NOT NULL REFERENCES search_space(id),
    workload_suite_id TEXT NOT NULL REFERENCES workload_suite(id),
    measurement_policy_id TEXT NOT NULL REFERENCES measurement_policy(id),
    placement_policy_json TEXT NOT NULL,
    baseline_json TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    frozen_at TEXT,
    completed_at TEXT,
    CHECK (status IN ('draft', 'planned', 'running', 'paused', 'completed', 'cancelled', 'failed'))
);

CREATE TABLE experiment_candidate (
    experiment_id TEXT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    generation_metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (experiment_id, candidate_id),
    UNIQUE (experiment_id, ordinal)
);

CREATE TABLE experiment_workload (
    experiment_id TEXT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    workload_case_id TEXT NOT NULL REFERENCES workload_case(id),
    suite_case_index INTEGER NOT NULL CHECK (suite_case_index >= 0),
    expansion_provenance_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (experiment_id, workload_case_id),
    UNIQUE (experiment_id, suite_case_index, workload_case_id)
);

CREATE TABLE resolved_placement (
    id TEXT PRIMARY KEY,
    placement_hash TEXT NOT NULL UNIQUE,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    host_id TEXT NOT NULL REFERENCES host(id),
    binary_id TEXT NOT NULL REFERENCES binary(id),
    production_context_size INTEGER NOT NULL CHECK (production_context_size > 0),
    n_gpu_layers INTEGER,
    n_cpu_moe INTEGER,
    split_mode TEXT,
    main_gpu INTEGER,
    tensor_split_json TEXT,
    override_tensor_json TEXT NOT NULL DEFAULT '[]',
    argv_json TEXT NOT NULL,
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT '',
    exit_code INTEGER,
    raw_result_json TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE benchmark_case (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    workload_case_id TEXT NOT NULL REFERENCES workload_case(id),
    placement_id TEXT REFERENCES resolved_placement(id),
    case_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'planned',
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    CHECK (status IN (
        'planned', 'running', 'completed', 'oom', 'timeout', 'invalid',
        'fit_failed', 'load_failed', 'benchmark_failed', 'parser_failed',
        'interrupted', 'cancelled'
    )),
    UNIQUE (experiment_id, ordinal)
);

CREATE TABLE benchmark_run (
    id TEXT PRIMARY KEY,
    benchmark_case_id TEXT NOT NULL REFERENCES benchmark_case(id),
    host_id TEXT NOT NULL REFERENCES host(id),
    binary_id TEXT NOT NULL REFERENCES binary(id),
    measurement_policy_id TEXT NOT NULL REFERENCES measurement_policy(id),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ns INTEGER CHECK (duration_ns IS NULL OR duration_ns >= 0),
    status TEXT NOT NULL,
    exit_code INTEGER,
    argv_json TEXT NOT NULL,
    environment_json TEXT NOT NULL,
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT '',
    raw_result_json TEXT,
    quality TEXT,
    quality_details_json TEXT,
    CHECK (status IN (
        'planned', 'running', 'completed', 'oom', 'timeout', 'invalid',
        'fit_failed', 'load_failed', 'benchmark_failed', 'parser_failed',
        'interrupted', 'cancelled'
    ))
);

CREATE TABLE benchmark_sample (
    run_id TEXT NOT NULL REFERENCES benchmark_run(id) ON DELETE CASCADE,
    sample_index INTEGER NOT NULL CHECK (sample_index >= 0),
    elapsed_ns INTEGER NOT NULL CHECK (elapsed_ns >= 0),
    tokens_per_second REAL NOT NULL CHECK (tokens_per_second >= 0),
    PRIMARY KEY (run_id, sample_index)
);

CREATE TABLE telemetry_sample (
    run_id TEXT NOT NULL REFERENCES benchmark_run(id) ON DELETE CASCADE,
    timestamp_ns INTEGER NOT NULL CHECK (timestamp_ns >= 0),
    cpu_system_pct REAL,
    cpu_user_pct REAL,
    cpu_system_mode_pct REAL,
    cpu_iowait_pct REAL,
    process_cpu_pct_normalized REAL,
    process_cpu_pct_raw REAL,
    process_user_time_ns INTEGER,
    process_system_time_ns INTEGER,
    process_threads INTEGER,
    cpu_freq_avg_hz INTEGER,
    cpu_freq_min_hz INTEGER,
    cpu_freq_max_hz INTEGER,
    cpu_temperature_c REAL,
    load_avg_1m REAL,
    load_avg_5m REAL,
    ram_used_bytes INTEGER,
    ram_available_bytes INTEGER,
    swap_used_bytes INTEGER,
    process_rss_bytes INTEGER,
    gpu_json TEXT NOT NULL DEFAULT '[]',
    cpu_per_core_json TEXT NOT NULL DEFAULT '[]',
    extra_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, timestamp_ns)
);

CREATE TABLE metric (
    run_id TEXT NOT NULL REFERENCES benchmark_run(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    value_real REAL,
    value_integer INTEGER,
    unit TEXT,
    dimensions_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, metric_name, dimensions_json),
    CHECK (value_real IS NOT NULL OR value_integer IS NOT NULL)
);

CREATE TABLE server_run (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiment(id),
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    placement_id TEXT REFERENCES resolved_placement(id),
    host_id TEXT NOT NULL REFERENCES host(id),
    server_binary_id TEXT NOT NULL REFERENCES binary(id),
    target_model_id TEXT NOT NULL,
    draft_model_id TEXT,
    spec_type TEXT,
    spec_draft_n_max INTEGER,
    argv_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ready_at TEXT,
    finished_at TEXT,
    status TEXT NOT NULL,
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT ''
);

CREATE TABLE server_benchmark (
    id TEXT PRIMARY KEY,
    server_run_id TEXT NOT NULL REFERENCES server_run(id) ON DELETE CASCADE,
    workload_case_id TEXT NOT NULL REFERENCES workload_case(id),
    avg_prompt_ts REAL,
    avg_pred_ts REAL,
    avg_latency_ms REAL,
    draft_n INTEGER,
    accepted_n INTEGER,
    accept_rate REAL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE candidate_evaluation (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    stage TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_run_case ON benchmark_run(benchmark_case_id);
CREATE INDEX idx_run_started ON benchmark_run(started_at);
CREATE INDEX idx_sample_run ON benchmark_sample(run_id);
CREATE INDEX idx_telemetry_run ON telemetry_sample(run_id, timestamp_ns);
CREATE INDEX idx_case_candidate ON benchmark_case(candidate_id);
CREATE INDEX idx_placement_candidate ON resolved_placement(candidate_id);
CREATE INDEX idx_experiment_candidate_candidate ON experiment_candidate(candidate_id);
CREATE INDEX idx_experiment_workload_workload ON experiment_workload(workload_case_id);
