import type {
  BinaryRecord,
  CandidateComparison,
  CandidateSummary,
  CandidateValidationHistory,
  Deployment,
  DeploymentCandidateItem,
  DeploymentParetoResult,
  DeploymentPlacement as DeploymentPlacementRecord,
  DeploymentPlanRequest,
  DeploymentPlanResponse,
  DeploymentProgress,
  DeploymentRun,
  DeploymentRunRequest,
  ExecutionRequest,
  Experiment,
  ExperimentProgress,
  LatencyEstimate,
  LauncherProfile,
  MatrixProjection,
  MetricDefinition,
  ModelRecord,
  ParameterDefinition,
  ParetoObjective,
  ParetoResult,
  Placement,
  PlanPreview,
  PlanSummary,
  ProfileListResponse,
  PromotionResponse,
  ResultRow,
  RunSummary,
  ServerValidationRequest
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // Keep status text when the body is not JSON.
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

function query(params: Array<[string, string | number | null | undefined]>): string {
  const value = new URLSearchParams();
  for (const [key, item] of params) {
    if (item !== null && item !== undefined && String(item) !== "") {
      value.append(key, String(item));
    }
  }
  const rendered = value.toString();
  return rendered ? "?" + rendered : "";
}

export const api = {
  profiles: () => request<ProfileListResponse>("/api/profiles"),
  profile: (id: string) => request<LauncherProfile>("/api/profiles/" + encodeURIComponent(id)),
  binaries: async () => (await request<{ items: BinaryRecord[] }>("/api/binaries")).items,
  parameters: async () =>
    (await request<{ items: ParameterDefinition[] }>("/api/parameters")).items,
  metrics: async () =>
    (await request<{ items: MetricDefinition[] }>("/api/metrics")).items,
  placements: async () =>
    (await request<{ items: Placement[] }>("/api/placements")).items,
  placement: (id: string) =>
    request<Placement>("/api/placements/" + encodeURIComponent(id)),
  models: async () =>
    (await request<{ items: ModelRecord[] }>("/api/models")).items,
  deployments: async () =>
    (await request<{ items: Deployment[] }>("/api/deployments")).items,
  deployment: (id: string) =>
    request<Deployment>("/api/deployments/" + encodeURIComponent(id)),
  createDeployment: (body: { deployment: unknown }) =>
    request<Deployment>("/api/deployments", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  planDeployment: (id: string, body: DeploymentPlanRequest) =>
    request<DeploymentPlanResponse>(
      "/api/deployments/" + encodeURIComponent(id) + "/plan",
      { method: "POST", body: JSON.stringify(body) }
    ),
  deploymentProgress: (id: string) =>
    request<DeploymentProgress>(
      "/api/deployments/" + encodeURIComponent(id) + "/progress"
    ),
  deploymentCandidates: async (id: string) =>
    (
      await request<{ items: DeploymentCandidateItem[] }>(
        "/api/deployments/" + encodeURIComponent(id) + "/candidates"
      )
    ).items,
  deploymentPlacements: async (id: string) =>
    (
      await request<{ items: DeploymentPlacementRecord[] }>(
        "/api/deployments/" + encodeURIComponent(id) + "/placements"
      )
    ).items,
  deploymentRuns: async (id: string) =>
    (
      await request<{ items: DeploymentRun[] }>(
        "/api/deployments/" + encodeURIComponent(id) + "/runs"
      )
    ).items,
  deploymentResults: async (id: string, filters: string[] = []) => {
    const params: Array<[string, string]> = [];
    filters.forEach((value) => params.push(["filter", value]));
    return (
      await request<{ deployment_id: string; rows: Record<string, unknown>[] }>(
        "/api/deployments/" + encodeURIComponent(id) + "/results" + query(params)
      )
    ).rows;
  },
  deploymentPareto: (
    id: string,
    options: {
      objectives: string[];
      constraints?: string[];
      filters?: string[];
    }
  ) => {
    const params: Array<[string, string]> = [];
    options.objectives.forEach((value) => params.push(["objective", value]));
    options.constraints?.forEach((value) => params.push(["constraint", value]));
    options.filters?.forEach((value) => params.push(["filter", value]));
    return request<DeploymentParetoResult>(
      "/api/deployments/" + encodeURIComponent(id) + "/pareto" + query(params)
    );
  },
  runDeployment: (
    id: string,
    body: DeploymentRunRequest,
    resume = false
  ) =>
    request<DeploymentProgress>(
      "/api/deployments/" +
        encodeURIComponent(id) +
        (resume ? "/resume" : "/run"),
      { method: "POST", body: JSON.stringify(body) }
    ),
  pauseDeployment: (id: string) =>
    request<DeploymentProgress>(
      "/api/deployments/" + encodeURIComponent(id) + "/pause",
      { method: "POST" }
    ),
  cancelDeployment: (id: string) =>
    request<DeploymentProgress>(
      "/api/deployments/" + encodeURIComponent(id) + "/cancel",
      { method: "POST" }
    ),
  experiments: async () =>
    (await request<{ items: Experiment[] }>("/api/experiments")).items,
  experiment: (id: string) =>
    request<Experiment>("/api/experiments/" + encodeURIComponent(id)),
  previewExperiment: (body: {
    base_candidate: unknown;
    search_space: unknown;
    workload_suite: unknown;
  }) =>
    request<PlanPreview>("/api/experiments/preview", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  createExperiment: (body: unknown) =>
    request<Experiment>("/api/experiments", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  planExperiment: (id: string) =>
    request<PlanSummary>("/api/experiments/" + encodeURIComponent(id) + "/plan", {
      method: "POST"
    }),
  progress: (id: string) =>
    request<ExperimentProgress>(
      "/api/experiments/" + encodeURIComponent(id) + "/progress"
    ),
  run: (id: string, body: ExecutionRequest, resume = false) =>
    request<ExperimentProgress>(
      "/api/experiments/" + encodeURIComponent(id) + (resume ? "/resume" : "/run"),
      { method: "POST", body: JSON.stringify(body) }
    ),
  pause: (id: string) =>
    request<ExperimentProgress>(
      "/api/experiments/" + encodeURIComponent(id) + "/pause",
      { method: "POST" }
    ),
  cancel: (id: string) =>
    request<ExperimentProgress>(
      "/api/experiments/" + encodeURIComponent(id) + "/cancel",
      { method: "POST" }
    ),
  candidates: async (id: string) =>
    (
      await request<{ items: CandidateSummary[] }>(
        "/api/experiments/" + encodeURIComponent(id) + "/candidates"
      )
    ).items,
  runs: async (id: string) =>
    (
      await request<{ items: RunSummary[] }>(
        "/api/experiments/" + encodeURIComponent(id) + "/runs"
      )
    ).items,
  results: async (
    id: string,
    filters: string[] = [],
    metrics: string[] = []
  ) => {
    const params: Array<[string, string]> = [];
    filters.forEach((value) => params.push(["filter", value]));
    metrics.forEach((value) => params.push(["metric", value]));
    return (
      await request<{ experiment_id: string; rows: ResultRow[] }>(
        "/api/experiments/" + encodeURIComponent(id) + "/results" + query(params)
      )
    ).rows;
  },
  matrix: (
    id: string,
    options: {
      x: string;
      y: string;
      metric: string;
      facet?: string | null;
      filters?: string[];
      qualities?: string[];
    }
  ) => {
    const params: Array<[string, string | null | undefined]> = [
      ["x", options.x],
      ["y", options.y],
      ["metric", options.metric],
      ["facet", options.facet]
    ];
    options.filters?.forEach((value) => params.push(["filter", value]));
    options.qualities?.forEach((value) => params.push(["quality", value]));
    return request<MatrixProjection>(
      "/api/experiments/" + encodeURIComponent(id) + "/matrix" + query(params)
    );
  },
  compare: (
    id: string,
    candidateId: string,
    metrics: string[],
    filters: string[] = []
  ) => {
    const params: Array<[string, string]> = [["candidate_id", candidateId]];
    metrics.forEach((value) => params.push(["metric", value]));
    filters.forEach((value) => params.push(["filter", value]));
    return request<CandidateComparison>(
      "/api/experiments/" + encodeURIComponent(id) + "/compare" + query(params)
    );
  },
  pareto: (
    id: string,
    objectives: ParetoObjective[],
    filters: string[] = []
  ) =>
    request<ParetoResult>(
      "/api/experiments/" + encodeURIComponent(id) + "/pareto",
      {
        method: "POST",
        body: JSON.stringify({ objectives, filters, qualities: [] })
      }
    ),
  latency: (
    id: string,
    candidateId: string,
    promptTokens: number,
    generateTokens: number,
    decodeStartDepthTokens?: number
  ) =>
    request<LatencyEstimate>(
      "/api/experiments/" +
        encodeURIComponent(id) +
        "/latency" +
        query([
          ["candidate_id", candidateId],
          ["prompt_tokens", promptTokens],
          ["generate_tokens", generateTokens],
          ["decode_start_depth_tokens", decodeStartDepthTokens]
        ])
    ),
  validationHistory: (experimentId: string, candidateId: string) =>
    request<CandidateValidationHistory>(
      "/api/experiments/" +
        encodeURIComponent(experimentId) +
        "/candidates/" +
        encodeURIComponent(candidateId) +
        "/validation"
    ),
  validateCandidate: (
    candidateId: string,
    body: ServerValidationRequest
  ) =>
    request<{
      experiment_id: string;
      candidate_id: string;
      server_run_id: string;
      benchmark_ids: string[];
      completed: boolean;
      speculative: boolean;
    }>("/api/candidates/" + encodeURIComponent(candidateId) + "/validate", {
      method: "POST",
      body: JSON.stringify(body)
    }),
  promoteCandidate: (
    candidateId: string,
    body: { experiment_id: string; source_profile_id?: string | null }
  ) =>
    request<PromotionResponse>(
      "/api/candidates/" + encodeURIComponent(candidateId) + "/promote",
      {
        method: "POST",
        body: JSON.stringify(body)
      }
    )
};

export function deploymentProgressEvents(
  deploymentId: string,
  onProgress: (progress: DeploymentProgress) => void,
  onError?: () => void
): () => void {
  const source = new EventSource(
    "/api/deployments/" + encodeURIComponent(deploymentId) + "/events"
  );
  source.addEventListener("progress", (event) => {
    onProgress(JSON.parse((event as MessageEvent<string>).data) as DeploymentProgress);
  });
  source.onerror = () => {
    source.close();
    onError?.();
  };
  return () => source.close();
}

export function progressEvents(
  experimentId: string,
  onProgress: (progress: ExperimentProgress) => void,
  onError?: () => void
): () => void {
  const source = new EventSource(
    "/api/experiments/" + encodeURIComponent(experimentId) + "/events"
  );
  source.addEventListener("progress", (event) => {
    onProgress(JSON.parse((event as MessageEvent<string>).data) as ExperimentProgress);
  });
  source.onerror = () => {
    source.close();
    onError?.();
  };
  return () => source.close();
}
