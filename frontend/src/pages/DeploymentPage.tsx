import { useEffect, useMemo, useState } from "react";
import { api, deploymentProgressEvents } from "../api";
import {
  EmptyState,
  ErrorBanner,
  JsonDetails,
  MetricCard,
  PageHeader,
  StatusBadge
} from "../components";
import type {
  BinaryRecord,
  Deployment,
  DeploymentCandidateItem,
  DeploymentMemoryMatrix,
  DeploymentParetoResult,
  DeploymentPlacement,
  DeploymentPlanRequest,
  DeploymentPlanResponse,
  DeploymentProgress,
  DeploymentRun,
  ModelRecord
} from "../types";

function resultNumber(row: Record<string, unknown>, key: string): number | null {
  const value = row[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function resultText(row: Record<string, unknown>, key: string): string | null {
  const value = row[key];
  return typeof value === "string" ? value : null;
}

function bytes(value: number | null): string {
  if (value === null) return "—";
  const gib = value / 1024 / 1024 / 1024;
  if (gib >= 1) return gib.toFixed(2) + " GiB";
  return (value / 1024 / 1024).toFixed(0) + " MiB";
}

interface PlannerDimensionDraft {
  key: string;
  path: string;
  values: string;
  condition: string;
}

const INSTANCE_DIMENSION_SUFFIXES = [
  "context.size",
  "context.cache_type_k",
  "context.cache_type_v",
  "compute.batch_size",
  "compute.ubatch_size",
  "compute.flash_attn",
  "placement.constraints.n_gpu_layers",
  "placement.constraints.split_mode",
  "placement.constraints.main_gpu",
  "placement.constraints.tensor_split",
  "requested_placement.devices",
  "requested_placement.n_gpu_layers",
  "requested_placement.split_mode",
  "requested_placement.main_gpu",
  "requested_placement.tensor_split",
  "requested_placement.override_tensor"
] as const;

function parseDimensionValues(raw: string): Array<string | number | boolean | null | Array<string | number | boolean | null>> {
  const parsed = JSON.parse(raw) as unknown;
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error("Dimension values must be a non-empty JSON array.");
  }
  for (const value of parsed) {
    if (
      value !== null &&
      typeof value !== "string" &&
      typeof value !== "number" &&
      typeof value !== "boolean" &&
      !(
        Array.isArray(value) &&
        value.every(
          (item) =>
            item === null ||
            typeof item === "string" ||
            typeof item === "number" ||
            typeof item === "boolean"
        )
      )
    ) {
      throw new Error("Dimension values may contain only JSON scalars or scalar arrays.");
    }
  }
  return parsed as Array<string | number | boolean | null | Array<string | number | boolean | null>>;
}

function MemoryMatrix({ matrix }: { matrix: DeploymentMemoryMatrix }) {
  return (
    <div className="candidate-table-wrap">
      <table className="data-table memory-matrix">
        <thead>
          <tr>
            <th>Instance / category</th>
            <th>Evidence</th>
            {matrix.devices.map((device) => (
              <th key={device}>{device}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row) => (
            <tr key={row.key + ":" + row.source}>
              <td><code>{row.key}</code></td>
              <td>
                <span className={"memory-source memory-source-" + row.source}>
                  {row.source}
                </span>
              </td>
              {matrix.devices.map((device) => (
                <td key={device}>{bytes(row.values[device] ?? null)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DeploymentPage({ deploymentId }: { deploymentId: string }) {
  const [deployment, setDeployment] = useState<Deployment | null>(null);
  const [progress, setProgress] = useState<DeploymentProgress | null>(null);
  const [candidates, setCandidates] = useState<DeploymentCandidateItem[]>([]);
  const [placements, setPlacements] = useState<DeploymentPlacement[]>([]);
  const [runs, setRuns] = useState<DeploymentRun[]>([]);
  const [results, setResults] = useState<Record<string, unknown>[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [binaries, setBinaries] = useState<BinaryRecord[]>([]);
  const [selectedPlacementId, setSelectedPlacementId] = useState("");
  const [plannerDimensions, setPlannerDimensions] = useState<PlannerDimensionDraft[]>([]);
  const [plannerConstraints, setPlannerConstraints] = useState("");
  const [helperByInstance, setHelperByInstance] = useState<Record<string, string>>({});
  const [planPreview, setPlanPreview] = useState<DeploymentPlanResponse | null>(null);
  const [plannerError, setPlannerError] = useState<unknown>(null);
  const [planning, setPlanning] = useState<"preview" | "plan" | null>(null);
  const [pareto, setPareto] = useState<DeploymentParetoResult | null>(null);
  const [paretoError, setParetoError] = useState<unknown>(null);
  const [paretoMetricA, setParetoMetricA] = useState("deployment.combined_tg_tps");
  const [paretoDirectionA, setParetoDirectionA] = useState<"max" | "min">("max");
  const [paretoPhaseA, setParetoPhaseA] = useState("dd");
  const [paretoMetricB, setParetoMetricB] = useState("deployment.combined_pp_tps");
  const [paretoDirectionB, setParetoDirectionB] = useState<"max" | "min">("max");
  const [paretoPhaseB, setParetoPhaseB] = useState("pp");
  const [retentionFloor, setRetentionFloor] = useState("");
  const [readinessTimeout, setReadinessTimeout] = useState(300);
  const [error, setError] = useState<unknown>(null);

  async function refresh() {
    try {
      const [
        deploymentValue,
        progressValue,
        candidateValues,
        placementValues,
        runValues,
        resultValues,
        modelValues,
        binaryValues
      ] = await Promise.all([
        api.deployment(deploymentId),
        api.deploymentProgress(deploymentId),
        api.deploymentCandidates(deploymentId),
        api.deploymentPlacements(deploymentId),
        api.deploymentRuns(deploymentId),
        api.deploymentResults(deploymentId),
        api.models(),
        api.binaries()
      ]);
      setDeployment(deploymentValue);
      setProgress(progressValue);
      setCandidates(candidateValues);
      setPlacements(placementValues);
      setRuns(runValues);
      setResults(resultValues);
      setModels(modelValues);
      setBinaries(binaryValues);
      const helpers = binaryValues.filter(
        (item) => item.kind === "llama-memory-estimator"
      );
      setHelperByInstance((current) => {
        const next = { ...current };
        for (const [index, instance] of deploymentValue.definition.instances.entries()) {
          if (!next[instance.instance_id] && helpers.length > 0) {
            next[instance.instance_id] = helpers[index]?.id ?? helpers[0].id;
          }
        }
        return next;
      });
      setPlannerDimensions((current) =>
        current.length > 0 || deploymentValue.definition.instances.length === 0
          ? current
          : [
              {
                key: "dimension-1",
                path:
                  "instances." +
                  deploymentValue.definition.instances[0].instance_id +
                  ".context.size",
                values: "[8192, 16384]",
                condition: ""
              }
            ]
      );
      setSelectedPlacementId((current) =>
        current || progressValue.current_placement_id || placementValues[0]?.id || ""
      );
    } catch (reason) {
      setError(reason);
    }
  }

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => {
      void Promise.all([
        api.deploymentProgress(deploymentId),
        api.deploymentRuns(deploymentId),
        api.deploymentPlacements(deploymentId),
        api.deploymentResults(deploymentId)
      ])
        .then(([progressValue, runValues, placementValues, resultValues]) => {
          setProgress(progressValue);
          setRuns(runValues);
          setPlacements(placementValues);
          setResults(resultValues);
        })
        .catch(() => undefined);
    }, 2500);
    return () => window.clearInterval(interval);
  }, [deploymentId]);

  useEffect(() => {
    const status = progress?.operation?.status;
    if (!status || !["running", "pausing", "cancelling"].includes(status)) {
      return;
    }
    return deploymentProgressEvents(
      deploymentId,
      (next) => {
        setProgress(next);
        void Promise.all([
          api.deploymentRuns(deploymentId),
          api.deploymentPlacements(deploymentId),
          api.deploymentResults(deploymentId)
        ]).then(([runValues, placementValues, resultValues]) => {
          setRuns(runValues);
          setPlacements(placementValues);
          setResults(resultValues);
        });
      },
      () => void refresh()
    );
  }, [deploymentId, progress?.operation?.id]);

  const selectedPlacement = useMemo(
    () => placements.find((item) => item.id === selectedPlacementId) ?? null,
    [placements, selectedPlacementId]
  );
  const memory = progress?.memory ?? selectedPlacement?.memory ?? null;
  const modelById = useMemo(
    () => new Map(models.map((item) => [item.id, item])),
    [models]
  );
  const runInputs = useMemo(() => {
    if (!deployment) return [];
    return deployment.definition.instances.map((instance) => ({
      instance_id: instance.instance_id,
      model_path: modelById.get(instance.model_artifact_id)?.files[0]?.path ?? ""
    }));
  }, [deployment, modelById]);
  const estimatorInputs = useMemo(() => {
    if (!deployment) return [];
    return deployment.definition.instances.map((instance) => ({
      instance_id: instance.instance_id,
      helper_binary_id: helperByInstance[instance.instance_id] ?? "",
      model_path: modelById.get(instance.model_artifact_id)?.files[0]?.path ?? ""
    }));
  }, [deployment, helperByInstance, modelById]);
  const memoryEstimatorBinaries = useMemo(
    () => binaries.filter((item) => item.kind === "llama-memory-estimator"),
    [binaries]
  );
  const missingModelPath = runInputs.some((item) => !item.model_path);
  const plannerUnavailable = estimatorInputs.some(
    (item) => !item.helper_binary_id || !item.model_path
  );
  const isActive =
    progress?.operation &&
    ["running", "pausing", "cancelling"].includes(progress.operation.status);
  const canResume = progress?.operation?.status === "paused";

  function updateDimension(index: number, patch: Partial<PlannerDimensionDraft>) {
    setPlannerDimensions((current) =>
      current.map((item, itemIndex) =>
        itemIndex === index ? { ...item, ...patch } : item
      )
    );
  }

  function addDimension() {
    const firstInstance = deployment?.definition.instances[0]?.instance_id ?? "model";
    setPlannerDimensions((current) => [
      ...current,
      {
        key: "dimension-" + Date.now(),
        path: "instances." + firstInstance + ".context.size",
        values: "[8192, 16384]",
        condition: ""
      }
    ]);
  }

  function buildPlanRequest(): DeploymentPlanRequest {
    if (plannerDimensions.length === 0) {
      throw new Error("Add at least one search dimension.");
    }
    if (plannerUnavailable) {
      throw new Error(
        "Each instance needs a registered model path and llama-memory-estimator binary."
      );
    }
    const constraints = plannerConstraints
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean)
      .map((expression) => ({ expression }));
    return {
      search_space: {
        schema: "llama-deployment-search-space",
        version: 1,
        dimensions: plannerDimensions.map((item) => ({
          path: item.path.trim(),
          values: parseDimensionValues(item.values),
          condition: item.condition.trim() || null
        })),
        constraints,
        strategy: { type: "grid" }
      },
      instances: estimatorInputs,
      timeout_seconds: 300
    };
  }

  async function previewPlan() {
    try {
      setPlannerError(null);
      setPlanning("preview");
      setPlanPreview(await api.previewDeployment(deploymentId, buildPlanRequest()));
    } catch (reason) {
      setPlanPreview(null);
      setPlannerError(reason);
    } finally {
      setPlanning(null);
    }
  }

  async function persistPlan() {
    try {
      setPlannerError(null);
      setPlanning("plan");
      const summary = await api.planDeployment(deploymentId, buildPlanRequest());
      setPlanPreview(summary);
      const [
        deploymentValue,
        progressValue,
        candidateValues,
        placementValues
      ] = await Promise.all([
        api.deployment(deploymentId),
        api.deploymentProgress(deploymentId),
        api.deploymentCandidates(deploymentId),
        api.deploymentPlacements(deploymentId)
      ]);
      setDeployment(deploymentValue);
      setProgress(progressValue);
      setCandidates(candidateValues);
      setPlacements(placementValues);
      if (summary.cases[0]?.deployment_placement_id) {
        setSelectedPlacementId(summary.cases[0].deployment_placement_id);
      } else if (placementValues[0]?.id) {
        setSelectedPlacementId(placementValues[0].id);
      }
    } catch (reason) {
      setPlannerError(reason);
    } finally {
      setPlanning(null);
    }
  }

  async function execute(resume: boolean) {
    if (!selectedPlacementId || missingModelPath) return;
    try {
      setError(null);
      setProgress(
        await api.runDeployment(
          deploymentId,
          {
            deployment_placement_id: selectedPlacementId,
            instances: runInputs,
            readiness_timeout_seconds: readinessTimeout
          },
          resume
        )
      );
    } catch (reason) {
      setError(reason);
    }
  }

  async function pause() {
    try {
      setProgress(await api.pauseDeployment(deploymentId));
    } catch (reason) {
      setError(reason);
    }
  }

  async function cancel() {
    try {
      setProgress(await api.cancelDeployment(deploymentId));
    } catch (reason) {
      setError(reason);
    }
  }

  async function loadPareto() {
    try {
      setParetoError(null);
      const objectives = [
        "x:" +
          paretoDirectionA +
          ":" +
          paretoMetricA +
          (paretoPhaseA ? "@workload.phase=" + paretoPhaseA : ""),
        "y:" +
          paretoDirectionB +
          ":" +
          paretoMetricB +
          (paretoPhaseB ? "@workload.phase=" + paretoPhaseB : "")
      ];
      const constraints =
        retentionFloor.trim() === ""
          ? []
          : ["deployment.min_retention:ge:" + retentionFloor.trim()];
      setPareto(
        await api.deploymentPareto(deploymentId, {
          objectives,
          constraints
        })
      );
    } catch (reason) {
      setPareto(null);
      setParetoError(reason);
    }
  }

  if (!deployment || !progress) {
    return (
      <main className="page">
        <ErrorBanner error={error} />
        <div className="skeleton large" />
      </main>
    );
  }

  return (
    <main className="page">
      <PageHeader
        eyebrow="Joint deployment"
        title={deployment.definition.instances
          .map((item) => item.role || item.instance_id)
          .join(" + ")}
        actions={
          <>
            <StatusBadge status={progress.deployment_status} />
            <a className="button" href="#/deployments">All deployments</a>
          </>
        }
      />
      <ErrorBanner error={error} />

      <section className="metric-grid">
        <MetricCard
          label="Plan"
          value={progress.planned_candidates}
          note={deployment.placement_count + " feasible placements"}
        />
        <MetricCard
          label="Completed"
          value={progress.completed_candidates}
          note={progress.failed_candidates + " failed"}
        />
        <MetricCard
          label="Combined prefill"
          value={
            progress.combined_prompt_tps === null
              ? "—"
              : progress.combined_prompt_tps.toFixed(1) + " t/s"
          }
          note={progress.current_workload_phase?.toUpperCase() ?? "No phase yet"}
        />
        <MetricCard
          label="Combined decode"
          value={
            progress.combined_decode_tps === null
              ? "—"
              : progress.combined_decode_tps.toFixed(1) + " t/s"
          }
          note="Aggregate overlap throughput"
        />
        <MetricCard
          label="Runs"
          value={deployment.run_count}
          note={progress.active_deployment_run ?? "No active run"}
        />
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Planning</div>
              <h2>Search joint placement space</h2>
              <p className="section-copy">
                Vary Candidate, placement, and memory-margin coordinates. Preview runs
                capability and memory feasibility without persisting plan cases; Plan
                stores the feasible placements and rejection history.
              </p>
            </div>
            <button className="button" type="button" onClick={addDimension}>
              Add dimension
            </button>
          </div>

          <div className="planner-estimators">
            {deployment.definition.instances.map((instance) => {
              const model = modelById.get(instance.model_artifact_id);
              return (
                <div className="planner-estimator-card" key={instance.instance_id}>
                  <strong>{instance.instance_id}</strong>
                  <div className="muted">{model?.files[0]?.path ?? "Model path unavailable"}</div>
                  <label className="field">
                    <span>Memory estimator</span>
                    <select
                      value={helperByInstance[instance.instance_id] ?? ""}
                      onChange={(event) =>
                        setHelperByInstance((current) => ({
                          ...current,
                          [instance.instance_id]: event.target.value
                        }))
                      }
                    >
                      <option value="">Choose…</option>
                      {memoryEstimatorBinaries.map((binary) => (
                        <option key={binary.id} value={binary.id}>
                          {binary.path}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              );
            })}
          </div>

          <div className="planner-dimension-list">
            {plannerDimensions.map((dimension, index) => (
              <div className="planner-dimension-row" key={dimension.key}>
                <label className="field">
                  <span>Search path</span>
                  <input
                    list={"deployment-paths-" + index}
                    value={dimension.path}
                    onChange={(event) =>
                      updateDimension(index, { path: event.target.value })
                    }
                  />
                  <datalist id={"deployment-paths-" + index}>
                    {deployment.definition.instances.flatMap((instance) =>
                      INSTANCE_DIMENSION_SUFFIXES.map((suffix) => (
                        <option
                          key={instance.instance_id + ":" + suffix}
                          value={"instances." + instance.instance_id + "." + suffix}
                        />
                      ))
                    )}
                    <option value="resource_policy.host_ram_margin_bytes" />
                  </datalist>
                </label>
                <label className="field">
                  <span>Values · JSON array</span>
                  <input
                    value={dimension.values}
                    onChange={(event) =>
                      updateDimension(index, { values: event.target.value })
                    }
                    placeholder='[8192, 16384] or [["CUDA0"], ["CUDA0","Vulkan0"]]'
                  />
                </label>
                <label className="field">
                  <span>Condition · optional</span>
                  <input
                    value={dimension.condition}
                    onChange={(event) =>
                      updateDimension(index, { condition: event.target.value })
                    }
                    placeholder="instances.model_a.compute.flash_attn == 'on'"
                  />
                </label>
                <button
                  className="icon-button planner-remove"
                  aria-label={"Remove search dimension " + (index + 1)}
                  type="button"
                  onClick={() =>
                    setPlannerDimensions((current) =>
                      current.filter((_, itemIndex) => itemIndex !== index)
                    )
                  }
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          <label className="field">
            <span>Constraints · one expression per line</span>
            <textarea
              className="planner-textarea"
              value={plannerConstraints}
              onChange={(event) => setPlannerConstraints(event.target.value)}
              placeholder="instances.model_a.context.size >= instances.model_b.context.size"
            />
          </label>

          <ErrorBanner error={plannerError} />
          {plannerUnavailable ? (
            <div className="banner banner-error">
              Planning requires a registered model file and llama-memory-estimator
              binary for every instance.
            </div>
          ) : null}

          <div className="button-row planner-actions">
            <button
              className="button"
              disabled={Boolean(planning) || plannerUnavailable}
              onClick={() => void previewPlan()}
            >
              {planning === "preview" ? "Previewing…" : "Preview plan"}
            </button>
            <button
              className="button button-primary"
              disabled={Boolean(planning) || plannerUnavailable}
              onClick={() => void persistPlan()}
            >
              {planning === "plan" ? "Planning…" : "Persist plan"}
            </button>
          </div>

          {planPreview ? (
            <div className="plan-preview deployment-plan-preview">
              <div>
                <div className="eyebrow">
                  {planPreview.plan_id ? "Persisted plan" : "Preview"}
                </div>
                <h2>{planPreview.valid_count} feasible placements</h2>
                <p>
                  {planPreview.raw_combinations} raw combinations ·{" "}
                  {planPreview.rejected_by_constraints} constraint-rejected ·{" "}
                  {planPreview.duplicate_candidates} duplicate
                </p>
              </div>
              <div className="plan-numbers">
                <div>
                  <strong>{planPreview.capability_rejected}</strong>
                  <span>capability rejected</span>
                </div>
                <div>
                  <strong>{planPreview.estimate_failed}</strong>
                  <span>estimate failed</span>
                </div>
                <div>
                  <strong>{planPreview.memory_rejected}</strong>
                  <span>memory rejected</span>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </section>

      <section className="panel live-panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Execution</div>
              <h2>{isActive ? "Deployment in progress" : "Run controls"}</h2>
              <p className="section-copy">
                Servers stay resident for synchronized DD/PP/PD/DP phases. Pause and
                cancel are durable and may be issued from another API or CLI process.
              </p>
            </div>
            {progress.current_deployment_candidate_id ? (
              <div className="current-work">
                <span>Candidate</span>
                <code>{progress.current_deployment_candidate_id}</code>
                <span>Placement</span>
                <code>{progress.current_placement_id ?? "—"}</code>
              </div>
            ) : null}
          </div>
          <div className="control-grid three">
            <label className="field grow">
              <span>Placement</span>
              <select
                value={selectedPlacementId}
                onChange={(event) => setSelectedPlacementId(event.target.value)}
                disabled={Boolean(isActive)}
              >
                <option value="">Choose a feasible placement…</option>
                {placements.map((item) => (
                  <option value={item.id} key={item.id}>
                    {item.id} · {item.deployment_candidate_id}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Readiness timeout (s)</span>
              <input
                type="number"
                min="1"
                value={readinessTimeout}
                onChange={(event) => setReadinessTimeout(Number(event.target.value))}
                disabled={Boolean(isActive)}
              />
            </label>
            <div className="button-row self-end">
              <button
                className="button button-primary"
                disabled={Boolean(isActive) || !selectedPlacementId || missingModelPath}
                onClick={() => void execute(false)}
              >
                Run
              </button>
              <button
                className="button"
                disabled={!canResume || missingModelPath}
                onClick={() => void execute(true)}
              >
                Resume
              </button>
              <button className="button" disabled={!isActive} onClick={() => void pause()}>
                Pause
              </button>
              <button
                className="button button-danger"
                disabled={!isActive && !canResume}
                onClick={() => void cancel()}
              >
                Cancel
              </button>
            </div>
          </div>
          {missingModelPath ? (
            <div className="banner banner-error">
              One or more deployment model artifacts have no registered file path.
              Register the model before running this placement.
            </div>
          ) : null}
          {progress.member_states.length > 0 ? (
            <div className="candidate-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Instance</th>
                    <th>Status</th>
                    <th>Endpoint</th>
                    <th>PID</th>
                    <th>Exit</th>
                  </tr>
                </thead>
                <tbody>
                  {progress.member_states.map((member) => (
                    <tr key={member.instance_id}>
                      <td><strong>{member.instance_id}</strong></td>
                      <td><StatusBadge status={member.status} /></td>
                      <td><code>{member.endpoint ?? "—"}</code></td>
                      <td>{member.pid ?? "—"}</td>
                      <td>{member.exit_code ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          {progress.failure_kind ? (
            <div className="banner banner-error">
              <strong>{progress.failure_kind}</strong>
              {progress.failure_details ? (
                <JsonDetails label="Failure details" value={progress.failure_details} />
              ) : null}
            </div>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Memory</div>
              <h2>Placement memory matrix</h2>
              <p className="section-copy">
                Projected estimator values and runtime telemetry are kept distinct.
              </p>
            </div>
          </div>
          {memory ? (
            <MemoryMatrix matrix={memory} />
          ) : (
            <EmptyState title="No placement memory yet.">
              Plan the deployment to populate projected device memory.
            </EmptyState>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Plan</div>
              <h2>Generated deployment Candidates</h2>
            </div>
          </div>
          {candidates.length === 0 ? (
            <EmptyState title="No planned Candidates yet.">
              Configure search dimensions above, preview pruning, then persist the plan.
            </EmptyState>
          ) : (
            <div className="candidate-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Candidate</th>
                    <th>Placements</th>
                    <th>Rejections</th>
                    <th>Generation</th>
                  </tr>
                </thead>
                <tbody>
                  {candidates.map((candidate) => (
                    <tr key={candidate.id}>
                      <td>
                        <a
                          className="text-link"
                          href={
                            "#/deployments/" +
                            deploymentId +
                            "/candidates/" +
                            candidate.id
                          }
                        >
                          <code>{candidate.id}</code>
                        </a>
                      </td>
                      <td>{candidate.placement_ids.length}</td>
                      <td>{candidate.rejection_count}</td>
                      <td><JsonDetails label="Coordinates" value={candidate.generation} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Concurrent results</div>
              <h2>DD / PP / PD / DP observations</h2>
              <p className="section-copy">
                Aggregate overlap throughput stays next to correctness and retention.
                Open a Candidate for per-instance interference evidence.
              </p>
            </div>
          </div>
          {results.length === 0 ? (
            <EmptyState title="No concurrent results yet." />
          ) : (
            <div className="candidate-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Phase</th>
                    <th>Candidate</th>
                    <th>Placement</th>
                    <th>Status</th>
                    <th>PP t/s</th>
                    <th>TG t/s</th>
                    <th>Min retention</th>
                  </tr>
                </thead>
                <tbody>
                  {results.slice().reverse().slice(0, 32).map((row, index) => {
                    const candidateId = resultText(row, "deployment_candidate_id");
                    const placementId = resultText(row, "deployment_placement_id");
                    const retention = resultNumber(row, "min_retention");
                    return (
                      <tr key={(resultText(row, "workload_run_id") ?? "row") + ":" + index}>
                        <td><strong>{(resultText(row, "phase") ?? "—").toUpperCase()}</strong></td>
                        <td>
                          {candidateId ? (
                            <a
                              className="text-link"
                              href={
                                "#/deployments/" +
                                deploymentId +
                                "/candidates/" +
                                candidateId
                              }
                            >
                              <code>{candidateId}</code>
                            </a>
                          ) : "—"}
                        </td>
                        <td><code>{placementId ?? "—"}</code></td>
                        <td>
                          <StatusBadge status={resultText(row, "workload_status") ?? "unknown"} />
                        </td>
                        <td>
                          {resultNumber(row, "combined_pp_tps")?.toFixed(2) ?? "—"}
                        </td>
                        <td>
                          {resultNumber(row, "combined_tg_tps")?.toFixed(2) ?? "—"}
                        </td>
                        <td>
                          {retention === null ? "—" : (retention * 100).toFixed(1) + "%"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Optimization</div>
              <h2>Deployment Pareto frontier</h2>
              <p className="section-copy">
                Constraints are applied before dominance. Failed or correctness-invalid
                placements remain in raw results but never enter the valid frontier.
              </p>
            </div>
          </div>
          <div className="pareto-controls">
            <div className="pareto-objective-card">
              <strong>Objective X</strong>
              <label className="field">
                <span>Metric</span>
                <select
                  aria-label="Deployment objective X metric"
                  value={paretoMetricA}
                  onChange={(event) => setParetoMetricA(event.target.value)}
                >
                  <option value="deployment.combined_tg_tps">Combined decode TPS</option>
                  <option value="deployment.combined_pp_tps">Combined prefill TPS</option>
                  <option value="deployment.min_retention">Minimum retention</option>
                  <option value="deployment.total_validated_context_tokens">Validated context</option>
                  <option value="deployment.min_device_headroom_bytes">Runtime headroom</option>
                  <option value="deployment.total_power_avg_w">Average power</option>
                </select>
              </label>
              <div className="control-grid two">
                <label className="field">
                  <span>Direction</span>
                  <select
                    value={paretoDirectionA}
                    onChange={(event) =>
                      setParetoDirectionA(event.target.value as "max" | "min")
                    }
                  >
                    <option value="max">Maximize</option>
                    <option value="min">Minimize</option>
                  </select>
                </label>
                <label className="field">
                  <span>Phase filter</span>
                  <select
                    value={paretoPhaseA}
                    onChange={(event) => setParetoPhaseA(event.target.value)}
                  >
                    <option value="">All phases</option>
                    {deployment.definition.workload_mix.phases.map((phase) => (
                      <option key={phase} value={phase}>{phase.toUpperCase()}</option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
            <div className="pareto-objective-card">
              <strong>Objective Y</strong>
              <label className="field">
                <span>Metric</span>
                <select
                  aria-label="Deployment objective Y metric"
                  value={paretoMetricB}
                  onChange={(event) => setParetoMetricB(event.target.value)}
                >
                  <option value="deployment.combined_tg_tps">Combined decode TPS</option>
                  <option value="deployment.combined_pp_tps">Combined prefill TPS</option>
                  <option value="deployment.min_retention">Minimum retention</option>
                  <option value="deployment.total_validated_context_tokens">Validated context</option>
                  <option value="deployment.min_device_headroom_bytes">Runtime headroom</option>
                  <option value="deployment.total_power_avg_w">Average power</option>
                </select>
              </label>
              <div className="control-grid two">
                <label className="field">
                  <span>Direction</span>
                  <select
                    value={paretoDirectionB}
                    onChange={(event) =>
                      setParetoDirectionB(event.target.value as "max" | "min")
                    }
                  >
                    <option value="max">Maximize</option>
                    <option value="min">Minimize</option>
                  </select>
                </label>
                <label className="field">
                  <span>Phase filter</span>
                  <select
                    value={paretoPhaseB}
                    onChange={(event) => setParetoPhaseB(event.target.value)}
                  >
                    <option value="">All phases</option>
                    {deployment.definition.workload_mix.phases.map((phase) => (
                      <option key={phase} value={phase}>{phase.toUpperCase()}</option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
          </div>
          <div className="control-grid two pareto-constraint-row">
            <label className="field">
              <span>Minimum retention constraint · optional</span>
              <input
                type="number"
                min="0"
                step="0.01"
                value={retentionFloor}
                placeholder="0.75"
                onChange={(event) => setRetentionFloor(event.target.value)}
              />
            </label>
            <div className="button-row self-end">
              <button className="button button-primary" onClick={() => void loadPareto()}>
                Calculate frontier
              </button>
            </div>
          </div>
          <ErrorBanner error={paretoError} />
          {pareto ? (
            pareto.result.frontier.length === 0 ? (
              <EmptyState title="No placement satisfies the selected objective evidence." />
            ) : (
              <div className="candidate-table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Candidate</th>
                      <th>Placement</th>
                      <th>X</th>
                      <th>Y</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pareto.result.frontier.map((point) => (
                      <tr key={point.deployment_placement_id}>
                        <td>
                          <a
                            className="text-link"
                            href={
                              "#/deployments/" +
                              deploymentId +
                              "/candidates/" +
                              point.deployment_candidate_id
                            }
                          >
                            <code>{point.deployment_candidate_id}</code>
                          </a>
                        </td>
                        <td><code>{point.deployment_placement_id}</code></td>
                        <td>{point.values.x?.toFixed(3) ?? "—"}</td>
                        <td>{point.values.y?.toFixed(3) ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          ) : null}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">History</div>
              <h2>Deployment runs</h2>
            </div>
          </div>
          {runs.length === 0 ? (
            <EmptyState title="No deployment runs yet." />
          ) : (
            <div className="candidate-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Status</th>
                    <th>Placement</th>
                    <th>Phases</th>
                    <th>Quality</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.slice().reverse().slice(0, 12).map((run) => (
                    <tr key={run.id}>
                      <td><code>{run.id}</code></td>
                      <td><StatusBadge status={run.status} /></td>
                      <td><code>{run.deployment_placement_id ?? "—"}</code></td>
                      <td>
                        {run.phases.map((phase) => phase.phase.toUpperCase()).join(" · ") || "—"}
                      </td>
                      <td>{run.quality ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <JsonDetails label="Deployment definition" value={deployment.definition} />
    </main>
  );
}
