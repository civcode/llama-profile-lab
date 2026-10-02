ALTER TABLE server_run ADD COLUMN target_model_path TEXT;
ALTER TABLE server_run ADD COLUMN draft_model_path TEXT;
ALTER TABLE server_run ADD COLUMN bind_host TEXT NOT NULL DEFAULT '127.0.0.1';
ALTER TABLE server_run ADD COLUMN bind_port INTEGER;
ALTER TABLE server_run ADD COLUMN duration_ns INTEGER;
ALTER TABLE server_run ADD COLUMN exit_code INTEGER;
ALTER TABLE server_run ADD COLUMN environment_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE server_benchmark ADD COLUMN speed_bench_binary_id TEXT REFERENCES binary(id);
ALTER TABLE server_benchmark ADD COLUMN category TEXT NOT NULL DEFAULT 'overall';
ALTER TABLE server_benchmark ADD COLUMN status TEXT NOT NULL DEFAULT 'completed';
ALTER TABLE server_benchmark ADD COLUMN argv_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE server_benchmark ADD COLUMN duration_ns INTEGER;
ALTER TABLE server_benchmark ADD COLUMN exit_code INTEGER;
ALTER TABLE server_benchmark ADD COLUMN stdout TEXT NOT NULL DEFAULT '';
ALTER TABLE server_benchmark ADD COLUMN stderr TEXT NOT NULL DEFAULT '';
ALTER TABLE server_benchmark ADD COLUMN requests INTEGER;
ALTER TABLE server_benchmark ADD COLUMN failed INTEGER;
ALTER TABLE server_benchmark ADD COLUMN turns INTEGER;

CREATE INDEX idx_server_run_experiment_candidate
ON server_run(experiment_id, candidate_id, started_at);

CREATE INDEX idx_server_benchmark_run
ON server_benchmark(server_run_id, created_at);

CREATE INDEX idx_candidate_evaluation_experiment_candidate
ON candidate_evaluation(experiment_id, candidate_id, created_at);
