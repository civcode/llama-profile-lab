CREATE INDEX idx_run_case_status
ON benchmark_run(benchmark_case_id, status);

CREATE INDEX idx_case_experiment_status
ON benchmark_case(experiment_id, status);
