import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  experiment: vi.fn(),
  progress: vi.fn(),
  candidates: vi.fn(),
  runs: vi.fn(),
  binaries: vi.fn(),
  profiles: vi.fn(),
  placements: vi.fn(),
  metrics: vi.fn(),
  matrix: vi.fn(),
  run: vi.fn(),
  pareto: vi.fn(),
  pause: vi.fn(),
  cancel: vi.fn(),
  progressEvents: vi.fn()
}));

vi.mock("../api", () => ({
  api: {
    experiment: mocks.experiment,
    progress: mocks.progress,
    candidates: mocks.candidates,
    runs: mocks.runs,
    binaries: mocks.binaries,
    profiles: mocks.profiles,
    placements: mocks.placements,
    metrics: mocks.metrics,
    matrix: mocks.matrix,
    run: mocks.run,
    pareto: mocks.pareto,
    pause: mocks.pause,
    cancel: mocks.cancel
  },
  progressEvents: mocks.progressEvents
}));

import { ExperimentPage } from "./ExperimentPage";

const candidate = {
  schema: "llama-profile-candidate",
  version: 1,
  model: {
    target_model_id: "launcher-profile:flash-128k",
    draft_model_id: null
  },
  context: {
    size: 131072,
    cache_type_k: "f16",
    cache_type_v: "f16",
    kv_offload: true,
    kv_unified: true
  },
  compute: {
    flash_attn: "on",
    batch_size: 4096,
    ubatch_size: 2048,
    threads: null,
    load_mode: "auto",
    lazy_mode: "auto",
    repack: true,
    no_host: false,
    no_op_offload: false
  },
  placement: {
    mode: "fit",
    fit: { target_mib: 256, min_context: 4096 },
    constraints: {
      n_gpu_layers: null,
      n_cpu_moe: 0,
      split_mode: "layer",
      main_gpu: 0,
      devices: "auto",
      tensor_split: null,
      override_tensor: []
    }
  },
  server: { parallel: 1 },
  speculative: { enabled: false, type: null, draft_n_max: null },
  extra_args: {}
};

const experiment = {
  id: "exp-ui",
  status: "planned",
  name: "Flash 128K sweep",
  base_candidate_id: "cand-base",
  search_space_id: "search",
  workload_suite_id: "suite",
  measurement_policy_id: "policy",
  created_at: "2026-01-01T00:00:00Z",
  frozen_at: "2026-01-01T00:00:00Z",
  completed_at: null,
  candidate_count: 1,
  workload_count: 1,
  benchmark_case_count: 1,
  incomplete_case_count: 1,
  definition: {
    name: "Flash 128K sweep",
    base_candidate_id: "cand-base",
    search_space_id: "search",
    workload_suite_id: "suite",
    measurement_policy_id: "policy",
    placement_policy: { type: "per-candidate" },
    baseline: { type: "base-candidate" }
  },
  base_candidate: candidate,
  search_space: {
    schema: "llama-search-space",
    version: 1,
    dimensions: [
      { path: "compute.batch_size", values: [4096] },
      { path: "compute.ubatch_size", values: [2048] }
    ],
    constraints: ["compute.ubatch_size <= compute.batch_size"],
    strategy: { type: "grid" }
  },
  workload_suite: {
    schema: "llama-workload-suite",
    version: 1,
    id: "suite",
    description: null,
    cases: [
      {
        kind: "microbench-decode",
        label: "TG256 @ 4K",
        safety_margin_tokens: 0,
        generate_tokens: 256,
        depth: { type: "absolute", tokens: 4096 }
      }
    ]
  },
  measurement_policy: {
    schema: "llama-measurement-policy",
    version: 1,
    warmup: true,
    repetitions: 3,
    delay_seconds: 0,
    adaptive: null
  }
};

const plannedProgress = {
  experiment_id: "exp-ui",
  experiment_status: "planned",
  total_cases: 1,
  completed_cases: 0,
  incomplete_cases: 1,
  case_status_counts: { planned: 1 },
  operation: null,
  current_candidate_id: null,
  current_candidate_ordinal: null,
  current_workload_case_id: null,
  current_suite_case_index: null,
  latest_run_id: null,
  latest_tokens_per_second: null,
  latest_metrics: {}
};

function commonMocks() {
  mocks.experiment.mockResolvedValue(experiment);
  mocks.progress.mockResolvedValue(plannedProgress);
  mocks.candidates.mockResolvedValue([
    {
      id: "cand-hidden",
      ordinal: 0,
      generation_metadata: {},
      candidate,
      workload_count: 1,
      benchmark_case_count: 1,
      completed_case_count: 0,
      server_validation_count: 0
    }
  ]);
  mocks.runs.mockResolvedValue([]);
  mocks.binaries.mockResolvedValue([
    {
      id: "bench",
      sha256: "a",
      kind: "llama-bench",
      path: "/opt/llama-bench",
      size_bytes: 1,
      mtime_ns: 1,
      git_commit: null,
      git_branch: null,
      git_dirty: null,
      build_number: null,
      build_info: {},
      capabilities: { options: ["--batch-size", "--ubatch-size"] },
      created_at: "2026-01-01T00:00:00Z"
    },
    {
      id: "fit",
      sha256: "b",
      kind: "llama-fit-params",
      path: "/opt/llama-fit-params",
      size_bytes: 1,
      mtime_ns: 1,
      git_commit: null,
      git_branch: null,
      git_dirty: null,
      build_number: null,
      build_info: {},
      capabilities: { options: ["--fit-target"] },
      created_at: "2026-01-01T00:00:00Z"
    }
  ]);
  mocks.profiles.mockResolvedValue({
    configured: true,
    source_path: "/profiles.json",
    items: [
      {
        id: "flash-128k",
        binary_key: "qwen38",
        binary_path: "/opt/llama-server",
        profiles: ["flash"],
        model_path: "/models/flash.gguf",
        draft_model_path: null,
        server_alias: "flash",
        args: {},
        candidate
      }
    ]
  });
  mocks.placements.mockResolvedValue([]);
  mocks.metrics.mockResolvedValue([
    { name: "throughput.median", label: "Median throughput", unit: "tokens/s" }
  ]);
  mocks.matrix.mockRejectedValue(new Error("no completed runs"));
  mocks.progressEvents.mockReturnValue(() => undefined);
}

describe("ExperimentPage execution controls", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    commonMocks();
  });

  it("starts a planned experiment through the HTTP client", async () => {
    mocks.run.mockResolvedValue({
      ...plannedProgress,
      experiment_status: "running",
      operation: {
        id: "op-1",
        experiment_id: "exp-ui",
        status: "running",
        started_at: "2026-01-01T00:00:01Z",
        finished_at: null,
        requested_action: null,
        summary: null,
        error: null
      }
    });

    render(<ExperimentPage experimentId="exp-ui" />);

    const start = await screen.findByRole("button", { name: "Start experiment" });
    expect(start).toBeEnabled();
    fireEvent.click(start);

    await waitFor(() =>
      expect(mocks.run).toHaveBeenCalledWith(
        "exp-ui",
        expect.objectContaining({
          binary_id: "bench",
          fit_binary_id: "fit",
          model_path: "/models/flash.gguf",
          telemetry_interval_ms: 1000
        }),
        false
      )
    );
  });

  it("shows human labels for an active operation and exposes pause/cancel", async () => {
    const active = {
      ...plannedProgress,
      experiment_status: "running",
      case_status_counts: { running: 1 },
      operation: {
        id: "op-2",
        experiment_id: "exp-ui",
        status: "running",
        started_at: "2026-01-01T00:00:01Z",
        finished_at: null,
        requested_action: null,
        summary: null,
        error: null
      },
      current_candidate_id: "cand-hidden",
      current_candidate_ordinal: 0,
      current_workload_case_id: "work-hidden",
      current_suite_case_index: 0,
      latest_run_id: "run-hidden",
      latest_tokens_per_second: 72.5,
      latest_metrics: {
        "telemetry.process_cpu_avg_pct_normalized": 55,
        "telemetry.cpu_system_avg_pct": 65,
        "telemetry.gpu_utilization_avg_pct": 91,
        "telemetry.gpu_vram_used_peak_bytes": 8589934592,
        "telemetry.ram_used_peak_bytes": 17179869184,
        "telemetry.gpu_temperature_peak_c": 70,
        "telemetry.cpu_temperature_peak_c": 62,
        "telemetry.gpu_power_avg_w": 210
      }
    };
    mocks.progress.mockResolvedValue(active);
    mocks.pause.mockResolvedValue({ ...active, experiment_status: "paused" });
    mocks.cancel.mockResolvedValue({ ...active, experiment_status: "cancelled" });

    render(<ExperimentPage experimentId="exp-ui" />);

    expect(await screen.findByText("Candidate #1")).toBeInTheDocument();
    expect(screen.getByText("TG256 @ 4K")).toBeInTheDocument();
    expect(screen.queryByText("cand-hidden")).not.toBeInTheDocument();
    expect(screen.queryByText("work-hidden")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    await waitFor(() => expect(mocks.pause).toHaveBeenCalledWith("exp-ui"));

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(mocks.cancel).toHaveBeenCalledWith("exp-ui"));
  });

  it("resumes a paused experiment without rerunning completed work", async () => {
    mocks.progress.mockResolvedValue({
      ...plannedProgress,
      experiment_status: "paused"
    });
    mocks.run.mockResolvedValue({
      ...plannedProgress,
      experiment_status: "running"
    });

    render(<ExperimentPage experimentId="exp-ui" />);

    const resume = await screen.findByRole("button", { name: "Resume experiment" });
    fireEvent.click(resume);

    await waitFor(() =>
      expect(mocks.run).toHaveBeenCalledWith(
        "exp-ui",
        expect.any(Object),
        true
      )
    );
  });
});
