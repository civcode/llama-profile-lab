ALTER TABLE deployment_workload_member
ADD COLUMN baseline_latency_ms REAL
CHECK (baseline_latency_ms IS NULL OR baseline_latency_ms >= 0);

ALTER TABLE deployment_workload_member
ADD COLUMN latency_increase_pct REAL;
