import { describe, expect, it } from "vitest";
import {
  candidateFromProfile,
  defaultWorkloads,
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
