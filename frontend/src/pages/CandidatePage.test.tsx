import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  experiment: vi.fn(),
  candidates: vi.fn(),
  results: vi.fn(),
  placements: vi.fn(),
  validationHistory: vi.fn(),
  profiles: vi.fn(),
  binaries: vi.fn(),
  compare: vi.fn(),
  latency: vi.fn(),
  validateCandidate: vi.fn(),
  promoteCandidate: vi.fn()
}));

vi.mock("../api", () => ({
  api: {
    experiment: mocks.experiment,
    candidates: mocks.candidates,
    results: mocks.results,
    placements: mocks.placements,
    validationHistory: mocks.validationHistory,
    profiles: mocks.profiles,
    binaries: mocks.binaries,
    compare: mocks.compare,
    latency: mocks.latency,
    validateCandidate: mocks.validateCandidate,
    promoteCandidate: mocks.promoteCandidate
  }
}));

import { CandidatePage } from "./CandidatePage";

const candidate = {
  schema: "llama-profile-candidate",
  version: 1,
  model: {
    target_model_id: "launcher-profile:demo",
    draft_model_id: null
  },
  context: {
    size: 8192,
    cache_type_k: "f16",
    cache_type_v: "f16",
    kv_offload: true,
    kv_unified: true
  },
  compute: {
    flash_attn: "on",
    batch_size: 4096,
    ubatch_size: 512,
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
  id: "exp-promotion",
  status: "completed",
  name: "Promotion acceptance",
  base_candidate_id: "cand-base",
  search_space_id: "search",
  workload_suite_id: "suite",
  measurement_policy_id: "policy",
  created_at: "2026-01-01T00:00:00Z",
  frozen_at: "2026-01-01T00:00:00Z",
  completed_at: "2026-01-01T00:10:00Z",
  candidate_count: 1,
  workload_count: 1,
  benchmark_case_count: 1,
  incomplete_case_count: 0,
  definition: {
    name: "Promotion acceptance",
    base_candidate_id: "cand-base",
    search_space_id: "search",
    workload_suite_id: "suite",
    measurement_policy_id: "policy",
    placement_policy: { type: "per-candidate" },
    baseline: { type: "base-candidate" }
  },
  base_candidate: {
    ...candidate,
    compute: { ...candidate.compute, batch_size: 2048 }
  },
  search_space: {
    schema: "llama-search-space",
    version: 1,
    dimensions: [{ path: "compute.batch_size", values: [2048, 4096] }],
    constraints: [],
    strategy: { type: "grid" }
  },
  workload_suite: {
    schema: "llama-workload-suite",
    version: 1,
    id: "suite",
    description: null,
    cases: [
      {
        kind: "microbench-prefill",
        label: "PP128",
        safety_margin_tokens: 0,
        prompt_tokens: 128,
        depth: { type: "absolute", tokens: 0 }
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

describe("CandidatePage promotion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.experiment.mockResolvedValue(experiment);
    mocks.candidates.mockResolvedValue([
      {
        id: "cand-promoted",
        ordinal: 0,
        generation_metadata: {},
        candidate,
        workload_count: 1,
        benchmark_case_count: 1,
        completed_case_count: 1,
        server_validation_count: 1
      }
    ]);
    mocks.results.mockResolvedValue([]);
    mocks.placements.mockResolvedValue([]);
    mocks.validationHistory.mockResolvedValue({
      experiment_id: "exp-promotion",
      candidate_id: "cand-promoted",
      evaluations: [
        {
          id: "eval-validation",
          stage: "server-validated",
          decision: "completed",
          reason: null,
          metrics: { server_run_id: "srv-1" },
          created_at: "2026-01-01T00:09:00Z"
        }
      ],
      benchmarks: []
    });
    mocks.profiles.mockResolvedValue({
      configured: true,
      source_path: "/launcher.json",
      items: [
        {
          id: "demo",
          binary_key: "custom",
          binary_path: "/opt/llama-server",
          profiles: ["flash"],
          model_path: "/models/demo.gguf",
          draft_model_path: null,
          server_alias: "demo",
          args: {
            "--ctx-size": 8192,
            "--batch-size": 2048,
            "--ubatch-size": 512,
            "--flash-attn": "on"
          },
          candidate: experiment.base_candidate
        }
      ]
    });
    mocks.binaries.mockResolvedValue([]);
    mocks.promoteCandidate.mockResolvedValue({
      id: "eval-promotion",
      experiment_id: "exp-promotion",
      candidate_id: "cand-promoted",
      source_profile: "demo",
      changes: [
        {
          path: "compute.batch_size",
          argument: "--batch-size",
          before: 2048,
          after: 4096
        }
      ],
      patch:
        '--- a/launcher-config.json\n+++ b/launcher-config.json\n-        "--batch-size": 2048,\n+        "--batch-size": 4096,\n',
      source_snapshot: {},
      proposed_snapshot: {},
      validation: {}
    });
  });

  it("generates a non-destructive patch after completed server validation", async () => {
    render(
      <CandidatePage
        experimentId="exp-promotion"
        candidateId="cand-promoted"
      />
    );

    const button = await screen.findByRole("button", {
      name: "Generate launcher patch"
    });
    expect(button).toBeEnabled();

    fireEvent.click(button);

    await waitFor(() =>
      expect(mocks.promoteCandidate).toHaveBeenCalledWith("cand-promoted", {
        experiment_id: "exp-promotion",
        source_profile_id: "demo"
      })
    );
    expect(await screen.findByText("eval-promotion")).toBeInTheDocument();
    expect(screen.getByText("demo")).toBeInTheDocument();
    expect(screen.getByText(/"--batch-size": 4096/)).toBeInTheDocument();
  });
});
