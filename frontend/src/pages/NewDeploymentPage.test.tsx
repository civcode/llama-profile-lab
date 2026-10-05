import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  experiments: vi.fn(),
  candidates: vi.fn(),
  models: vi.fn(),
  binaries: vi.fn(),
  createDeployment: vi.fn()
}));

vi.mock("../api", () => ({
  api: {
    experiments: mocks.experiments,
    candidates: mocks.candidates,
    models: mocks.models,
    binaries: mocks.binaries,
    createDeployment: mocks.createDeployment
  }
}));

import { NewDeploymentPage } from "./NewDeploymentPage";

const candidate = {
  id: "cand-ui",
  ordinal: 0,
  generation_metadata: {},
  candidate: {
    schema: "llama-profile-candidate",
    version: 1,
    model: { target_model_id: "model", draft_model_id: null },
    context: {
      size: 8192,
      cache_type_k: "f16",
      cache_type_v: "f16",
      kv_offload: true,
      kv_unified: true
    },
    compute: {
      flash_attn: "on",
      batch_size: 1024,
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
  },
  workload_count: 1,
  benchmark_case_count: 1,
  completed_case_count: 1,
  server_validation_count: 0
};

const experiment = {
  id: "exp-ui",
  status: "completed",
  name: "Reference sweep",
  base_candidate_id: "cand-ui",
  search_space_id: "search",
  workload_suite_id: "suite-ui",
  measurement_policy_id: "policy",
  created_at: "2026-10-05T00:00:00Z",
  frozen_at: "2026-10-05T00:00:00Z",
  completed_at: "2026-10-05T01:00:00Z",
  candidate_count: 1,
  workload_count: 1,
  benchmark_case_count: 1,
  incomplete_case_count: 0,
  definition: {
    name: "Reference sweep",
    base_candidate_id: "cand-ui",
    search_space_id: "search",
    workload_suite_id: "suite-ui",
    measurement_policy_id: "policy",
    placement_policy: { type: "per-candidate" },
    baseline: { type: "base-candidate" }
  },
  base_candidate: candidate.candidate,
  search_space: {
    schema: "llama-search-space",
    version: 1,
    dimensions: [{ path: "context.size", values: [8192] }],
    constraints: [],
    strategy: { type: "grid" }
  },
  workload_suite: {
    schema: "llama-workload-suite",
    version: 1,
    id: "suite-ui",
    description: null,
    cases: [
      {
        kind: "microbench-decode",
        generate_tokens: 128,
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

describe("NewDeploymentPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.experiments.mockResolvedValue([experiment]);
    mocks.candidates.mockResolvedValue([candidate]);
    mocks.models.mockResolvedValue([
      {
        id: "model-a",
        identity_hash: "a",
        architecture: "qwen",
        parameter_count: 100,
        quantization: "Q4",
        size_bytes: 1000,
        metadata: {},
        created_at: "2026-10-05T00:00:00Z",
        files: [
          {
            id: "file-a",
            part_index: 0,
            path: "/models/a.gguf",
            sha256: "a",
            size_bytes: 1000
          }
        ]
      },
      {
        id: "model-b",
        identity_hash: "b",
        architecture: "qwen",
        parameter_count: 50,
        quantization: "Q4",
        size_bytes: 500,
        metadata: {},
        created_at: "2026-10-05T00:00:00Z",
        files: [
          {
            id: "file-b",
            part_index: 0,
            path: "/models/b.gguf",
            sha256: "b",
            size_bytes: 500
          }
        ]
      }
    ]);
    mocks.binaries.mockResolvedValue([
      {
        id: "server-a",
        sha256: "s",
        kind: "llama-server",
        path: "/opt/llama-server",
        size_bytes: 1,
        mtime_ns: 1,
        git_commit: null,
        git_branch: null,
        git_dirty: null,
        build_number: null,
        build_info: {},
        capabilities: {},
        created_at: "2026-10-05T00:00:00Z"
      }
    ]);
    mocks.createDeployment.mockResolvedValue({ id: "deploy-created" });
    window.location.hash = "#/deployments/new";
  });

  it("creates a deployment from persisted candidates, models, and binaries", async () => {
    render(<NewDeploymentPage />);

    const create = await screen.findByRole("button", { name: "Create deployment" });
    expect(create).toBeEnabled();
    fireEvent.click(create);

    await waitFor(() => expect(mocks.createDeployment).toHaveBeenCalledTimes(1));
    const payload = mocks.createDeployment.mock.calls[0][0];
    expect(payload.deployment.instances).toHaveLength(2);
    expect(payload.deployment.instances[0]).toMatchObject({
      instance_id: "model_a",
      candidate_id: "cand-ui",
      model_artifact_id: "model-a",
      binary_id: "server-a"
    });
    expect(payload.deployment.instances[1]).toMatchObject({
      instance_id: "model_b",
      candidate_id: "cand-ui",
      model_artifact_id: "model-b",
      binary_id: "server-a"
    });
    expect(payload.deployment.workload_mix).toEqual({
      workload_suite_id: "suite-ui",
      phases: ["dd", "pp", "pd", "dp"]
    });
    await waitFor(() =>
      expect(window.location.hash).toBe("#/deployments/deploy-created")
    );
  });

  it("rejects a non-positive total power limit before submission", async () => {
    render(<NewDeploymentPage />);

    const power = await screen.findByLabelText("Maximum total power (W)");
    fireEvent.change(power, { target: { value: "0" } });

    expect(screen.getByRole("button", { name: "Create deployment" })).toBeDisabled();
  });
});
