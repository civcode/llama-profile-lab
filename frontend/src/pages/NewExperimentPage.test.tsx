import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  createExperiment: vi.fn(),
  planExperiment: vi.fn(),
  profiles: vi.fn(),
  parameters: vi.fn(),
  binaries: vi.fn(),
  placements: vi.fn()
}));

vi.mock("../api", () => ({
  api: {
    createExperiment: mocks.createExperiment,
    planExperiment: mocks.planExperiment,
    profiles: mocks.profiles,
    parameters: mocks.parameters,
    binaries: mocks.binaries,
    placements: mocks.placements
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
    mocks.createExperiment.mockResolvedValue({ id: "exp-ui" });
    mocks.planExperiment.mockResolvedValue({ experiment_id: "exp-ui" });
    window.location.hash = "#/new";
  });

  it("builds and plans the reference 11 × 4 experiment without JSON editing", async () => {
    render(<NewExperimentPage />);
    expect(await screen.findByText("11 valid candidates")).toBeInTheDocument();
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
