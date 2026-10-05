import { describe, expect, it } from "vitest";
import { parseRoute } from "./App";

describe("hash routing", () => {
  it("routes experiment and candidate URLs without a router dependency", () => {
    expect(parseRoute("#/")).toEqual({ kind: "experiments" });
    expect(parseRoute("#/new")).toEqual({ kind: "new" });
    expect(parseRoute("#/deployments")).toEqual({ kind: "deployments" });
    expect(parseRoute("#/deployments/new")).toEqual({ kind: "newDeployment" });
    expect(parseRoute("#/deployments/deploy-1")).toEqual({
      kind: "deployment",
      deploymentId: "deploy-1"
    });
    expect(parseRoute("#/deployments/deploy-1/candidates/cand-2")).toEqual({
      kind: "deploymentCandidate",
      deploymentId: "deploy-1",
      candidateId: "cand-2"
    });
    expect(parseRoute("#/experiments/exp-1")).toEqual({
      kind: "experiment",
      experimentId: "exp-1"
    });
    expect(parseRoute("#/experiments/exp-1/compare")).toEqual({
      kind: "comparison",
      experimentId: "exp-1"
    });
    expect(parseRoute("#/experiments/exp-1/candidates/cand-2")).toEqual({
      kind: "candidate",
      experimentId: "exp-1",
      candidateId: "cand-2"
    });
  });
});
