import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  deployment: vi.fn(),
  deploymentProgress: vi.fn(),
  deploymentCandidates: vi.fn(),
  deploymentPlacements: vi.fn(),
  deploymentRuns: vi.fn(),
  deploymentResults: vi.fn(),
  deploymentPareto: vi.fn(),
  models: vi.fn(),
  binaries: vi.fn(),
  previewDeployment: vi.fn(),
  planDeployment: vi.fn(),
  runDeployment: vi.fn(),
  pauseDeployment: vi.fn(),
  cancelDeployment: vi.fn(),
  deploymentProgressEvents: vi.fn()
}));

vi.mock("../api", () => ({
  api: {
    deployment: mocks.deployment,
    deploymentProgress: mocks.deploymentProgress,
    deploymentCandidates: mocks.deploymentCandidates,
    deploymentPlacements: mocks.deploymentPlacements,
    deploymentRuns: mocks.deploymentRuns,
    deploymentResults: mocks.deploymentResults,
    deploymentPareto: mocks.deploymentPareto,
    models: mocks.models,
    binaries: mocks.binaries,
    previewDeployment: mocks.previewDeployment,
    planDeployment: mocks.planDeployment,
    runDeployment: mocks.runDeployment,
    pauseDeployment: mocks.pauseDeployment,
    cancelDeployment: mocks.cancelDeployment
  },
  deploymentProgressEvents: mocks.deploymentProgressEvents
}));

import { DeploymentPage } from "./DeploymentPage";

const deployment = {
  id: "deploy-ui",
  status: "draft",
  created_at: "2026-10-05T12:00:00Z",
  plan_count: 0,
  placement_count: 0,
  run_count: 0,
  latest_plan_id: null,
  definition: {
    schema: "llama-profile-deployment-candidate",
    version: 1,
    instances: [
      {
        instance_id: "qwen",
        candidate_id: "cand-qwen",
        role: "primary",
        model_artifact_id: "model-qwen",
        binary_id: "server-qwen",
        requested_placement: {
          devices: null,
          n_gpu_layers: null,
          split_mode: null,
          main_gpu: null,
          tensor_split: null,
          override_tensor: []
        },
        server_identity: "qwen"
      },
      {
        instance_id: "flash",
        candidate_id: "cand-flash",
        role: "secondary",
        model_artifact_id: "model-flash",
        binary_id: "server-flash",
        requested_placement: {
          devices: null,
          n_gpu_layers: null,
          split_mode: null,
          main_gpu: null,
          tensor_split: null,
          override_tensor: []
        },
        server_identity: "flash"
      }
    ],
    resource_policy: {
      device_memory_margin_bytes: {},
      logical_device_mappings: [],
      host_ram_margin_bytes: 0,
      allow_cpu_offload: false,
      allow_swap: false,
      allowed_devices: [],
      allowed_backend_pairs: [],
      maximum_total_power_w: null
    },
    workload_mix: {
      workload_suite_id: "suite-ui",
      phases: ["dd", "pp", "pd", "dp"]
    }
  }
};

const progress = {
  deployment_id: "deploy-ui",
  deployment_status: "draft",
  planned_candidates: 0,
  completed_candidates: 0,
  failed_candidates: 0,
  active_deployment_run: null,
  current_deployment_candidate_id: null,
  current_placement_id: null,
  member_states: [],
  current_workload_phase: null,
  combined_prompt_tps: null,
  combined_decode_tps: null,
  memory: null,
  failure_kind: null,
  failure_details: null,
  operation: null
};

const models = [
  {
    id: "model-qwen",
    identity_hash: "qwen",
    architecture: "qwen",
    parameter_count: 100,
    quantization: "Q4",
    size_bytes: 1000,
    metadata: {},
    created_at: "2026-10-05T00:00:00Z",
    files: [
      {
        id: "file-qwen",
        part_index: 0,
        path: "/models/qwen.gguf",
        sha256: "a",
        size_bytes: 1000
      }
    ]
  },
  {
    id: "model-flash",
    identity_hash: "flash",
    architecture: "qwen",
    parameter_count: 50,
    quantization: "Q4",
    size_bytes: 500,
    metadata: {},
    created_at: "2026-10-05T00:00:00Z",
    files: [
      {
        id: "file-flash",
        part_index: 0,
        path: "/models/flash.gguf",
        sha256: "b",
        size_bytes: 500
      }
    ]
  }
];

const binaries = [
  {
    id: "estimator",
    sha256: "e",
    kind: "llama-memory-estimator",
    path: "/opt/llama-memory-estimator",
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
];

describe("DeploymentPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.deployment.mockResolvedValue(deployment);
    mocks.deploymentProgress.mockResolvedValue(progress);
    mocks.deploymentCandidates.mockResolvedValue([]);
    mocks.deploymentPlacements.mockResolvedValue([]);
    mocks.deploymentRuns.mockResolvedValue([]);
    mocks.deploymentResults.mockResolvedValue([]);
    mocks.deploymentPareto.mockResolvedValue({
      deployment_id: "deploy-ui",
      result: {
        objectives: [],
        constraints: [],
        evaluated_count: 1,
        frontier: [
          {
            deployment_candidate_id: "deploy-candidate-ui",
            deployment_placement_id: "place-ui",
            values: { x: 80, y: 120 }
          }
        ],
        excluded: {}
      }
    });
    mocks.models.mockResolvedValue(models);
    mocks.binaries.mockResolvedValue(binaries);
    mocks.deploymentProgressEvents.mockReturnValue(() => undefined);
    mocks.previewDeployment.mockResolvedValue({
      base_deployment_candidate_id: "deploy-ui",
      host_id: "host-ui",
      raw_combinations: 2,
      rejected_by_constraints: 0,
      duplicate_candidates: 0,
      symmetry_reduced: 0,
      capability_rejected: 0,
      estimate_failed: 0,
      memory_rejected: 1,
      valid_count: 1,
      plan_id: null,
      cases: []
    });
  });

  it("previews the M4 search space with model paths and estimator binaries", async () => {
    render(<DeploymentPage deploymentId="deploy-ui" />);

    const preview = await screen.findByRole("button", { name: "Preview plan" });
    expect(preview).toBeEnabled();
    fireEvent.click(preview);

    await waitFor(() =>
      expect(mocks.previewDeployment).toHaveBeenCalledWith(
        "deploy-ui",
        expect.objectContaining({
          search_space: expect.objectContaining({
            dimensions: [
              expect.objectContaining({
                path: "instances.qwen.context.size",
                values: [8192, 16384]
              })
            ]
          }),
          instances: [
            {
              instance_id: "qwen",
              helper_binary_id: "estimator",
              model_path: "/models/qwen.gguf"
            },
            {
              instance_id: "flash",
              helper_binary_id: "estimator",
              model_path: "/models/flash.gguf"
            }
          ]
        })
      )
    );
    expect(await screen.findByText("1 feasible placements")).toBeInTheDocument();
    expect(screen.getByText("memory rejected")).toBeInTheDocument();
  });

  it("builds a constrained deployment Pareto request", async () => {
    render(<DeploymentPage deploymentId="deploy-ui" />);

    expect(await screen.findByText("Deployment Pareto frontier")).toBeInTheDocument();
    fireEvent.change(
      screen.getByLabelText("Minimum retention constraint · optional"),
      { target: { value: "0.75" } }
    );
    fireEvent.click(screen.getByRole("button", { name: "Calculate frontier" }));

    await waitFor(() =>
      expect(mocks.deploymentPareto).toHaveBeenCalledWith("deploy-ui", {
        objectives: [
          "x:max:deployment.combined_tg_tps@workload.phase=dd",
          "y:max:deployment.combined_pp_tps@workload.phase=pp"
        ],
        constraints: ["deployment.min_retention:ge:0.75"]
      })
    );
  });

  it("renders projected and runtime memory evidence distinctly", async () => {
    mocks.deploymentPlacements.mockResolvedValue([
      {
        id: "place-ui",
        deployment_candidate_id: "deploy-candidate-ui",
        host_id: "host-ui",
        feasibility: "feasible",
        placement: {},
        memory: {
          deployment_placement_id: "place-ui",
          deployment_run_id: "run-ui",
          devices: ["GPU0"],
          rows: [
            { key: "qwen.model", source: "projected", values: { GPU0: 1024 } },
            { key: "runtime_peak", source: "runtime", values: { GPU0: 900 } }
          ]
        }
      }
    ]);

    render(<DeploymentPage deploymentId="deploy-ui" />);

    expect(await screen.findByText("projected")).toBeInTheDocument();
    expect(screen.getByText("runtime")).toBeInTheDocument();
    expect(screen.getByText("qwen.model")).toBeInTheDocument();
  });
});
