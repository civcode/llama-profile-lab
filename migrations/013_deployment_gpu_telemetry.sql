CREATE TABLE deployment_gpu_sample (
    deployment_run_id TEXT NOT NULL
        REFERENCES deployment_run(id) ON DELETE CASCADE,
    timestamp_ns INTEGER NOT NULL CHECK (timestamp_ns >= 0),
    gpu_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (deployment_run_id, timestamp_ns)
);

CREATE INDEX idx_deployment_gpu_sample_run
ON deployment_gpu_sample(deployment_run_id, timestamp_ns);
