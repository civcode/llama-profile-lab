export type JsonScalar = string | number | boolean | null;

export interface LauncherProfile {
  id: string;
  binary_key: string;
  binary_path: string;
  profiles: string[];
  model_path: string;
  draft_model_path: string | null;
  server_alias: string | null;
  args: Record<string, JsonScalar>;
  candidate: Candidate;
}

export interface ProfileListResponse {
  configured: boolean;
  source_path: string | null;
  items: LauncherProfile[];
}

export interface BinaryRecord {
  id: string;
  sha256: string;
  kind: string;
  path: string;
  size_bytes: number;
  mtime_ns: number;
  git_commit: string | null;
  git_branch: string | null;
  git_dirty: boolean | null;
  build_number: string | null;
  build_info: Record<string, unknown>;
  capabilities: Record<string, unknown>;
  created_at: string;
}

export interface ParameterDefinition {
  path: string;
  label: string;
  category: string;
  value_types: string[];
  cli_argument: string | null;
  affects_placement: boolean;
  supported_by: string[];
  minimum: number | null;
  maximum: number | null;
  string_choices: string[] | null;
}

export interface MetricDefinition {
  name: string;
  label: string;
  unit: string;
}

export interface Candidate {
  schema: "llama-profile-candidate";
  version: 1;
  model: {
    target_model_id: string;
    draft_model_id: string | null;
  };
  context: {
    size: number;
    cache_type_k: string;
    cache_type_v: string;
    kv_offload: boolean;
    kv_unified: boolean;
  };
  compute: {
    flash_attn: "on" | "off" | "auto";
    batch_size: number;
    ubatch_size: number;
    threads: number | null;
    load_mode: string;
    lazy_mode: string;
    repack: boolean;
    no_host: boolean;
    no_op_offload: boolean;
  };
  placement: {
    mode: "fit" | "fixed";
    fit: { target_mib: number; min_context: number } | null;
    constraints: {
      n_gpu_layers: number | "auto" | "all" | null;
      n_cpu_moe: number;
      split_mode: string;
      main_gpu: number;
      devices: "auto" | string[];
      tensor_split: number[] | null;
      override_tensor: string[];
    };
  };
  server: { parallel: number };
  speculative: {
    enabled: boolean;
    type: string | null;
    draft_n_max: number | null;
  };
  extra_args: Record<string, JsonScalar | JsonScalar[]>;
}

export interface SearchDimension {
  path: string;
  values: JsonScalar[];
  condition?: string | null;
}

export interface SearchSpace {
  schema: "llama-search-space";
  version: 1;
  dimensions: SearchDimension[];
  constraints: string[];
  strategy: { type: "grid" };
}

export type DepthExpression =
  | { type: "absolute"; tokens: number }
  | { type: "fraction"; value: number };

export type WorkloadSuiteCase =
  | {
      kind: "microbench-prefill";
      label?: string | null;
      safety_margin_tokens?: number;
      prompt_tokens: number;
      depth: DepthExpression;
    }
  | {
      kind: "microbench-decode";
      label?: string | null;
      safety_margin_tokens?: number;
      generate_tokens: number;
      depth: DepthExpression;
    }
  | {
      kind: "microbench-combined";
      label?: string | null;
      safety_margin_tokens?: number;
      prompt_tokens: number;
      generate_tokens: number;
      depth: DepthExpression;
    }
  | {
      kind: "speed-bench";
      label?: string | null;
      safety_margin_tokens?: number;
      speed_bench: {
        bench: string;
        categories: string[];
        output_tokens: number;
        concurrency: number;
        limit: number | null;
        request: Record<string, JsonScalar>;
      };
    };

export interface WorkloadSuite {
  schema: "llama-workload-suite";
  version: 1;
  id: string;
  description: string | null;
  cases: WorkloadSuiteCase[];
}

export interface MeasurementPolicy {
  schema: "llama-measurement-policy";
  version: 1;
  warmup: boolean;
  repetitions: number | null;
  delay_seconds: number;
  adaptive: null | {
    minimum_repetitions: number;
    maximum_repetitions: number;
    target_relative_error: number;
  };
}

export interface ExperimentDefinition {
  name: string;
  base_candidate_id: string;
  search_space_id: string;
  workload_suite_id: string;
  measurement_policy_id: string;
  placement_policy:
    | { type: "per-candidate" }
    | { type: "fixed"; placement_id: string };
  baseline:
    | { type: "base-candidate" }
    | { type: "candidate"; candidate_id: string };
}

export interface Experiment {
  id: string;
  status: string;
  name: string;
  base_candidate_id: string;
  search_space_id: string;
  workload_suite_id: string;
  measurement_policy_id: string;
  created_at: string;
  frozen_at: string | null;
  completed_at: string | null;
  candidate_count: number;
  workload_count: number;
  benchmark_case_count: number;
  incomplete_case_count: number;
  definition: ExperimentDefinition;
  base_candidate: Candidate;
  search_space: SearchSpace;
  workload_suite: WorkloadSuite;
  measurement_policy: MeasurementPolicy;
}

export interface PlanPreview {
  raw_combinations: number;
  rejected_by_constraints: number;
  duplicate_candidates: number;
  candidate_count: number;
  workloads_per_candidate: number | null;
  benchmark_case_count: number;
  unique_workload_count: number;
}

export interface PlanSummary {
  experiment_id: string;
  experiment_name: string;
  raw_combinations: number;
  rejected_by_constraints: number;
  duplicate_candidates: number;
  candidate_count: number;
  workloads_per_candidate: number | null;
  benchmark_case_count: number;
  unique_workload_count: number;
}

export interface ExecutionRequest {
  binary_id: string;
  model_path: string;
  fit_binary_id: string | null;
  timeout_seconds?: number | null;
  fit_timeout_seconds?: number | null;
  limit?: number | null;
  telemetry_interval_ms: number;
}

export interface ExperimentProgress {
  experiment_id: string;
  experiment_status: string;
  total_cases: number;
  completed_cases: number;
  incomplete_cases: number;
  case_status_counts: Record<string, number>;
  operation: {
    id: string;
    experiment_id: string;
    status: string;
    started_at: string;
    finished_at: string | null;
    requested_action: "pause" | "cancel" | null;
    summary: {
      experiment_id: string;
      attempted: number;
      completed: number;
      failed: number;
      remaining: number;
      interrupted: boolean;
      limited: boolean;
    } | null;
    error: string | null;
  } | null;
  current_candidate_id: string | null;
  current_candidate_ordinal: number | null;
  current_workload_case_id: string | null;
  current_suite_case_index: number | null;
  latest_run_id: string | null;
  latest_tokens_per_second: number | null;
  latest_metrics: Record<string, number>;
}

export interface CandidateSummary {
  id: string;
  ordinal: number;
  generation_metadata: Record<string, unknown>;
  candidate: Candidate;
  workload_count: number;
  benchmark_case_count: number;
  completed_case_count: number;
  server_validation_count: number;
}

export interface RunSummary {
  id: string;
  benchmark_case_id: string;
  candidate_id: string;
  workload_case_id: string;
  placement_id: string | null;
  workload_kind: string;
  host_id: string;
  binary_id: string;
  measurement_policy_id: string;
  started_at: string;
  finished_at: string | null;
  duration_ns: number | null;
  status: string;
  exit_code: number | null;
  quality: string | null;
  quality_details: Record<string, unknown> | null;
}

export interface ResultRow {
  experiment_id: string;
  candidate_id: string;
  candidate_ordinal: number;
  suite_case_index: number;
  workload_case_id: string;
  workload_kind: string;
  prompt_tokens: number | null;
  generate_tokens: number | null;
  depth_tokens: number | null;
  run_count: number;
  qualities: string[];
  [key: string]: unknown;
}

export interface MatrixCell {
  x: JsonScalar;
  y: JsonScalar;
  value: number;
  candidate_id: string;
  workload_case_id: string;
  run_count: number;
  sample_count: number;
}

export interface MatrixProjection {
  experiment_id: string;
  x_path: string;
  y_path: string;
  metric: string;
  facet_path: string | null;
  x_values: JsonScalar[];
  y_values: JsonScalar[];
  facets: { value: JsonScalar; cells: MatrixCell[] }[];
}

export interface Placement {
  id: string;
  candidate_id: string;
  host_id: string;
  binary_id: string;
  fit_attempt_id: string | null;
  production_context_size: number;
  n_gpu_layers: number;
  n_cpu_moe: number;
  split_mode: string;
  main_gpu: number;
  devices: string | string[];
  tensor_split: number[] | null;
  override_tensor: string[];
  request: Record<string, unknown>;
  created_at: string;
}

export interface CandidateComparison {
  experiment_id: string;
  candidate_id: string;
  baseline_candidate_id: string;
  deltas: {
    suite_case_index: number;
    workload_label: string;
    metric: string;
    baseline_value: number | null;
    candidate_value: number | null;
    delta: number | null;
    percent_delta: number | null;
  }[];
}

export interface ParetoObjective {
  key: string;
  direction: "maximize" | "minimize";
  metric: string;
  filters: { path: string; value: JsonScalar }[];
}

export interface ParetoResult {
  experiment_id: string;
  objectives: ParetoObjective[];
  evaluated_count: number;
  frontier: {
    candidate_id: string;
    candidate_ordinal: number;
    values: Record<string, number>;
  }[];
  excluded: Record<string, string>;
}

export interface LatencyEstimate {
  experiment_id: string;
  candidate_id: string;
  prompt_tokens: number;
  generate_tokens: number;
  decode_start_depth_tokens: number;
  prefill_tokens_per_second: number;
  decode_average_tokens_per_second: number;
  prefill_seconds: number;
  decode_seconds: number;
  total_seconds: number;
  prefill_curve: { tokens: number; tokens_per_second: number }[];
  decode_curve: { tokens: number; tokens_per_second: number }[];
}

export interface CandidateValidationHistory {
  experiment_id: string;
  candidate_id: string;
  evaluations: {
    id: string;
    stage: string;
    decision: string;
    reason: string | null;
    metrics: Record<string, unknown>;
    created_at: string;
  }[];
  benchmarks: {
    id: string;
    server_run_id: string;
    workload_case_id: string;
    category: string;
    status: string;
    requests: number | null;
    failed: number | null;
    turns: number | null;
    avg_prompt_ts: number | null;
    avg_pred_ts: number | null;
    avg_latency_ms: number | null;
    draft_n: number | null;
    accepted_n: number | null;
    accept_rate: number | null;
    created_at: string;
  }[];
}

export interface PromotionResponse {
  id: string;
  experiment_id: string;
  candidate_id: string;
  source_profile: string;
  changes: {
    path: string;
    argument: string;
    before: JsonScalar;
    after: JsonScalar;
  }[];
  patch: string;
  source_snapshot: Record<string, unknown>;
  proposed_snapshot: Record<string, unknown>;
  validation: Record<string, unknown>;
}

export interface ServerValidationRequest {
  experiment_id: string;
  server_binary_id: string;
  speed_bench_binary_id: string;
  model_path: string;
  draft_model_path?: string | null;
  placement_id?: string | null;
  model_name?: string | null;
  host?: string;
  port?: number;
  readiness_timeout_seconds?: number;
  request_timeout_seconds?: number;
  benchmark_timeout_seconds?: number | null;
  workload_case_id?: string | null;
}
