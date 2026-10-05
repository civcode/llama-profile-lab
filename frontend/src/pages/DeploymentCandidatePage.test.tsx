import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  deploymentCandidates: vi.fn(),
  deploymentPlacements: vi.fn(),
  deploymentRuns: vi.fn(),
  deploymentResults: vi.fn(),
  deployment: vi.fn(),
  experiments: vi.fn(),
  candidates: vi.fn(),
  promoteDeployment: vi.fn()
}));

vi.mock("../api", () => ({
  api: {
    deploymentCandidates: mocks.deploymentCandidates,
    deploymentPlacements: mocks.deploymentPlacements,
    deploymentRuns: mocks.deploymentRuns,
    deploymentResults: mocks.deploymentResults,
    deployment: mocks.deployment,
    experiments: mocks.experiments,
    candidates: mocks.candidates,
    promoteDeployment: mocks.promoteDeployment
  }
}));

import { DeploymentCandidatePage } from "./DeploymentCandidatePage";

describe("DeploymentCandidatePage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.deploymentCandidates.mockResolvedValue([
      {
        id: "deploy-candidate",
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
            workload_suite_id: "suite",
            phases: ["dd", "pp"]
          }
        },
        generation: {
          assignments: {
            "instances.qwen.context.size": 8192
          }
        },
        rejection_count: 1,
        rejections: [
          {
            id: "reject-1",
            stage: "memory",
            reason: "device_memory_exceeded",
            details: { device_id: "GPU0" },
            created_at: "2026-10-05T00:00:00Z"
          }
        ],
        placement_ids: ["place-1"]
      }
    ]);
    mocks.deploymentPlacements.mockResolvedValue([
      {
        id: "place-1",
        deployment_candidate_id: "deploy-candidate",
        host_id: "host",
        feasibility: "feasible",
        placement: {},
        memory: {
          deployment_placement_id: "place-1",
          deployment_run_id: "run-1",
          devices: ["GPU0"],
          rows: [
            {
              key: "projected_free",
              source: "projected",
              values: { GPU0: 1073741824 }
            }
          ]
        }
      }
    ]);
    mocks.deploymentRuns.mockResolvedValue([
      {
        id: "run-1",
        deployment_candidate_id: "deploy-candidate",
        deployment_placement_id: "place-1",
        status: "completed",
        quality: "clean",
        failure_kind: null,
        started_at: "2026-10-05T00:00:00Z",
        finished_at: "2026-10-05T00:01:00Z",
        duration_ns: 1,
        members: [],
        phases: []
      }
    ]);
    mocks.deployment.mockResolvedValue({
      id: "deploy-root",
      status: "completed",
      created_at: "2026-10-05T00:00:00Z",
      plan_count: 1,
      placement_count: 1,
      run_count: 1,
      latest_plan_id: "plan-1",
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
          workload_suite_id: "suite",
          phases: ["dd", "pp"]
        }
      }
    });
    mocks.experiments.mockResolvedValue([
      { id: "exp-qwen", name: "Qwen source" },
      { id: "exp-flash", name: "Flash source" }
    ]);
    mocks.candidates.mockImplementation((experimentId: string) =>
      Promise.resolve([
        {
          id: experimentId === "exp-qwen" ? "cand-qwen" : "cand-flash"
        }
      ])
    );
    mocks.promoteDeployment.mockResolvedValue({
      id: "deploypromo-1",
      base_deployment_candidate_id: "deploy-root",
      deployment_candidate_id: "deploy-candidate",
      deployment_placement_id: "place-1",
      sources: [
        {
          instance_id: "qwen",
          experiment_id: "exp-qwen",
          source_candidate_id: "cand-qwen",
          candidate_id: "cand-qwen-final",
          source_profile_id: "qwen"
        },
        {
          instance_id: "flash",
          experiment_id: "exp-flash",
          source_candidate_id: "cand-flash",
          candidate_id: "cand-flash-final",
          source_profile_id: "flash"
        }
      ],
      changes: [
        {
          instance_id: "qwen",
          candidate_id: "cand-qwen-final",
          source_profile_id: "qwen",
          changes: [
            {
              path: "placement.resolved.devices",
              argument: "--device",
              before: "auto",
              after: "CUDA0,Vulkan0"
            }
          ]
        },
        {
          instance_id: "flash",
          candidate_id: "cand-flash-final",
          source_profile_id: "flash",
          changes: []
        }
      ],
      patch: "--- a/launcher-config.json\n+++ b/launcher-config.json\n+ CUDA0,Vulkan0",
      source_snapshot: {},
      proposed_snapshot: {},
      evidence: { deployment_run_id: "run-1" }
    });
    mocks.deploymentResults.mockResolvedValue([
      {
        deployment_candidate_id: "deploy-candidate",
        deployment_placement_id: "place-1",
        deployment_run_id: "run-1",
        workload_run_id: "work-1",
        deployment_status: "completed",
        workload_status: "completed",
        correctness_valid: true,
        phase: "dd",
        combined_tg_tps: 100,
        combined_pp_tps: null,
        min_retention: 0.8,
        "instance.qwen.mode": "decode",
        "instance.qwen.standalone_tps": 80,
        "instance.qwen.overlap_tps": 64,
        "instance.qwen.retention": 0.8,
        "instance.qwen.latency_ms": 12,
        "instance.qwen.latency_increase_pct": 20,
        "instance.flash.mode": "decode",
        "instance.flash.standalone_tps": 50,
        "instance.flash.overlap_tps": 45,
        "instance.flash.retention": 0.9,
        "instance.flash.latency_ms": 10,
        "instance.flash.latency_increase_pct": 5
      }
    ]);
  });

  it("generates one coordinated proposal for all source profiles", async () => {
    render(
      <DeploymentCandidatePage
        deploymentId="deploy-root"
        candidateId="deploy-candidate"
      />
    );

    const button = await screen.findByRole("button", {
      name: "Generate coordinated proposal"
    });
    expect(button).toBeEnabled();
    fireEvent.click(button);

    await waitFor(() =>
      expect(mocks.promoteDeployment).toHaveBeenCalledWith("deploy-root", {
        deployment_placement_id: "place-1",
        sources: [
          { instance_id: "qwen", experiment_id: "exp-qwen" },
          { instance_id: "flash", experiment_id: "exp-flash" }
        ]
      })
    );
    expect(await screen.findByText("Unified launcher patch")).toBeInTheDocument();
    expect(screen.getByText(/CUDA0,Vulkan0/)).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("shows rejection explanations and per-instance interference evidence", async () => {
    render(
      <DeploymentCandidatePage
        deploymentId="deploy-root"
        candidateId="deploy-candidate"
      />
    );

    expect(await screen.findByText("device memory exceeded")).toBeInTheDocument();
    expect(screen.getByText("Standalone TPS")).toBeInTheDocument();
    expect(screen.getByText("Concurrent TPS")).toBeInTheDocument();
    expect(screen.getByText("80.00")).toBeInTheDocument();
    expect(screen.getByText("64.00")).toBeInTheDocument();
    expect(screen.getByText("80.00%")).toBeInTheDocument();
    expect(screen.getByText("1.00 GiB")).toBeInTheDocument();
  });
});
