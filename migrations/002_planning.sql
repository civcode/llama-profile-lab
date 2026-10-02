ALTER TABLE experiment_workload RENAME TO experiment_workload_legacy;

CREATE TABLE experiment_workload (
    experiment_id TEXT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL REFERENCES candidate(id),
    workload_case_id TEXT NOT NULL REFERENCES workload_case(id),
    suite_case_index INTEGER NOT NULL CHECK (suite_case_index >= 0),
    expansion_provenance_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (experiment_id, candidate_id, workload_case_id)
);

INSERT OR IGNORE INTO experiment_workload(
    experiment_id,
    candidate_id,
    workload_case_id,
    suite_case_index,
    expansion_provenance_json
)
SELECT
    legacy.experiment_id,
    COALESCE(benchmark_case.candidate_id, experiment.base_candidate_id),
    legacy.workload_case_id,
    legacy.suite_case_index,
    legacy.expansion_provenance_json
FROM experiment_workload_legacy AS legacy
JOIN experiment ON experiment.id = legacy.experiment_id
LEFT JOIN benchmark_case
    ON benchmark_case.experiment_id = legacy.experiment_id
   AND benchmark_case.workload_case_id = legacy.workload_case_id;

DROP TABLE experiment_workload_legacy;

CREATE INDEX idx_experiment_workload_workload
ON experiment_workload(workload_case_id);

CREATE INDEX idx_experiment_workload_candidate
ON experiment_workload(experiment_id, candidate_id);
