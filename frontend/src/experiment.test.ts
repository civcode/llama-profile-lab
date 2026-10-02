import { describe, expect, it } from "vitest";
import {
  candidateFromProfile,
  defaultWorkloads,
  dimensionSupported,
  previewCandidateCount
} from "./experiment";
import type { LauncherProfile } from "./types";

const profile: LauncherProfile = {
  id: "flash-128k",
  binary_key: "qwen38",
  binary_path: "/opt/llama-server",
  profiles: ["qwen-base", "flash-attn", "flash-next"],
  model_path: "/models/flash.gguf",
  draft_model_path: null,
  server_alias: "flash",
  args: {
    "--ctx-size": 131072,
    "--batch-size": 4096,
    "--ubatch-size": 2048,
    "--flash-attn": "on",
    "--load-mode": "mmap",
    "--lazy-mode": "on",
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
      load_mode: "mmap",
      lazy_mode: "on",
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
  }
};

describe("experiment planning helpers", () => {
  it("maps a launcher profile into an immutable Candidate", () => {
    const candidate = candidateFromProfile(profile);
    expect(candidate.context.size).toBe(131072);
    expect(candidate.compute.batch_size).toBe(4096);
    expect(candidate.compute.ubatch_size).toBe(2048);
    expect(candidate.compute.flash_attn).toBe("on");
    expect(candidate.placement.fit?.target_mib).toBe(256);
    expect(candidate.model.target_model_id).toBe("launcher-profile:flash-128k");
  });

  it("previews the reference batch/ubatch sweep as eleven valid candidates", () => {
    const candidate = candidateFromProfile(profile);
    const result = previewCandidateCount(
      candidate,
      [
        { path: "compute.batch_size", values: [2048, 4096, 8192] },
        { path: "compute.ubatch_size", values: [512, 1024, 2048, 4096] }
      ],
      ["compute.ubatch_size <= compute.batch_size"]
    );
    expect(result).toEqual({ raw: 12, valid: 11, rejected: 1 });
    expect(defaultWorkloads()).toHaveLength(4);
    expect(result.valid * defaultWorkloads().length).toBe(44);
  });
});

describe("binary capability support", () => {
  const definition = {
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
  };
  const bench = {
    id: "bench",
    sha256: "x",
    kind: "llama-bench",
    path: "/bin/llama-bench",
    size_bytes: 1,
    mtime_ns: 1,
    git_commit: null,
    git_branch: null,
    git_dirty: null,
    build_number: null,
    build_info: {},
    capabilities: { options: ["--batch-size", "--ubatch-size"] },
    created_at: "2026-01-01T00:00:00Z"
  };

  it("disables parameters unsupported by the selected tool kind", () => {
    expect(dimensionSupported(definition, bench)).toBe(false);
  });

  it("disables required CLI options missing from the selected binary", () => {
    expect(
      dimensionSupported(
        {
          ...definition,
          path: "compute.batch_size",
          supported_by: ["llama-bench"],
          cli_argument: "--batch-size"
        },
        {
          ...bench,
          capabilities: { options: ["--ubatch-size"] }
        }
      )
    ).toBe(false);
  });
});

