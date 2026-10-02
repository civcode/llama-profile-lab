import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  createExperiment: vi.fn(),
  planExperiment: vi.fn(),
  profiles: vi.fn(),
  parameters: vi.fn(),
  binaries: vi.fn(),
  placements: vi.fn(),
  previewExperiment: vi.fn()
}));

vi.mock("../api", () => ({
  api: {
    createExperiment: mocks.createExperiment,
    planExperiment: mocks.planExperiment,
    profiles: mocks.profiles,
    parameters: mocks.parameters,
    binaries: mocks.binaries,
    placements: mocks.placements,
    previewExperiment: mocks.previewExperiment
  }
}));

import { NewExperimentPage } from "./NewExperimentPage";

describe("NewExperimentPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.profiles.mockResolvedValue({
      configured: true,
      source_path: "/profiles.json",
      items: [
        {
          id: "flash-128k",
          binary_key: "qwen38",
          binary_path: "/opt/llama-server",
          profiles: ["flash-attn"],
          model_path: "/models/flash.gguf",
          draft_model_path: null,
          server_alias: "flash",
          args: {
            "--ctx-size": 131072,
            "--batch-size": 4096,
            "--ubatch-size": 2048,
            "--flash-attn": "on",
            "--fit-target": 256
          },
          candidate: {
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
            speculative: {
              enabled: false,
              type: null,
              draft_n_max: null
            },
            extra_args: {}
          }
        }
      ]
    });
    mocks.parameters.mockResolvedValue([
      {
        path: "compute.batch_size",
        label: "Batch size",
        category: "Compute",
        value_types: ["int"],
        cli_argument: "--batch-size",
        affects_placement: true,
        supported_by: ["llama-bench"],
        minimum: 1,
        maximum: null,
        string_choices: null
      },
      {
        path: "compute.ubatch_size",
        label: "Physical batch size",
        category: "Compute",
        value_types: ["int"],
        cli_argument: "--ubatch-size",
        affects_placement: true,
        supported_by: ["llama-bench"],
        minimum: 1,
        maximum: null,
        string_choices: null
      },
      {
        path: "speculative.draft_n_max",
        label: "Maximum draft tokens",
        category: "Speculative decoding",
        value_types: ["int"],
        cli_argument: "--spec-draft-n-max",
        affects_placement: false,
        supported_by: ["llama-server"],
        minimum: 1,
        maximum: null,
        string_choices: null
      }
    ]);
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
      }
    ]);
    mocks.placements.mockResolvedValue([]);
    mocks.previewExperiment.mockImplementation((body) => {
      const speedCases = body.workload_suite.cases.filter(
        (item: { kind: string }) => item.kind === "speed-bench"
      ).length;
      return Promise.resolve({
        raw_combinations: 12,
        rejected_by_constraints: 1,
        duplicate_candidates: 0,
        candidate_count: 11,
        workloads_per_candidate: 4 + speedCases,
        benchmark_case_count: 44,
        unique_workload_count: 4 + speedCases
      });
    });
    mocks.createExperiment.mockResolvedValue({ id: "exp-ui" });
    mocks.planExperiment.mockResolvedValue({ experiment_id: "exp-ui" });
    window.location.hash = "#/new";
  });

  it("builds and plans the reference 11 × 4 experiment without JSON editing", async () => {
    render(<NewExperimentPage />);
    expect(await screen.findByText("11 valid candidates")).toBeInTheDocument();
    expect(mocks.previewExperiment).toHaveBeenCalled();
    expect(screen.getByText("44")).toBeInTheDocument();
    expect(screen.getByText("benchmark cases")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Save + plan experiment" }));

    await waitFor(() => expect(mocks.createExperiment).toHaveBeenCalledTimes(1));
    const payload = mocks.createExperiment.mock.calls[0][0];
    expect(payload.search_space.dimensions).toHaveLength(2);
    expect(payload.search_space.constraints).toEqual([
      "compute.ubatch_size <= compute.batch_size"
    ]);
    expect(payload.workload_suite.cases).toHaveLength(4);
    expect(payload.measurement_policy.repetitions).toBe(3);
    await waitFor(() => expect(mocks.planExperiment).toHaveBeenCalledWith("exp-ui"));
  });



  it("removes an enabled dimension when the selected binary cannot execute it", async () => {
    render(<NewExperimentPage />);
    expect(await screen.findByText("11 valid candidates")).toBeInTheDocument();

    const capabilityTarget = screen.getByLabelText("Capability target");
    fireEvent.change(capabilityTarget, { target: { value: "" } });

    const speculative = screen.getByRole("checkbox", {
      name: /Maximum draft tokens/
    });
    expect(speculative).toBeEnabled();
    fireEvent.click(speculative);
    fireEvent.change(screen.getByLabelText("Maximum draft tokens values"), {
      target: { value: "8" }
    });
    expect(speculative).toBeChecked();

    fireEvent.change(capabilityTarget, { target: { value: "bench" } });

    await waitFor(() => expect(speculative).not.toBeChecked());
    expect(speculative).toBeDisabled();
    expect(screen.getByText("Not supported by llama-bench")).toBeInTheDocument();

    await waitFor(() => {
      const body = mocks.previewExperiment.mock.calls.at(-1)?.[0];
      expect(
        body.search_space.dimensions.some(
          (item: { path: string }) => item.path === "speculative.draft_n_max"
        )
      ).toBe(false);
    });
  });

  it("adds a server validation workload without inflating microbenchmark case count", async () => {
    render(<NewExperimentPage />);
    expect(await screen.findByText("11 valid candidates")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Add server validation" })
    );
    expect(screen.getByText("server workload")).toBeInTheDocument();
    expect(screen.getByText("44")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Save + plan experiment" })
    );

    await waitFor(() => expect(mocks.createExperiment).toHaveBeenCalledTimes(1));
    const payload = mocks.createExperiment.mock.calls[0][0];
    expect(payload.workload_suite.cases).toHaveLength(5);
    expect(payload.workload_suite.cases[4]).toMatchObject({
      kind: "speed-bench",
      speed_bench: {
        bench: "throughput_1k",
        categories: ["all"],
        output_tokens: 256,
        concurrency: 1
      }
    });
    await waitFor(() => expect(mocks.planExperiment).toHaveBeenCalledWith("exp-ui"));
  });
});
