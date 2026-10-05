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
  Deployment,
  DeploymentCandidateItem,
  DeploymentMemoryMatrix,
  DeploymentPlacement,
  DeploymentProgress,
  DeploymentRun,
  ModelRecord
} from "../types";

function bytes(value: number | null): string {
  if (value === null) return "—";
  const gib = value / 1024 / 1024 / 1024;
  if (gib >= 1) return gib.toFixed(2) + " GiB";
  return (value / 1024 / 1024).toFixed(0) + " MiB";
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
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [selectedPlacementId, setSelectedPlacementId] = useState("");
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
        modelValues
      ] = await Promise.all([
        api.deployment(deploymentId),
        api.deploymentProgress(deploymentId),
        api.deploymentCandidates(deploymentId),
        api.deploymentPlacements(deploymentId),
        api.deploymentRuns(deploymentId),
        api.models()
      ]);
      setDeployment(deploymentValue);
      setProgress(progressValue);
      setCandidates(candidateValues);
      setPlacements(placementValues);
      setRuns(runValues);
      setModels(modelValues);
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
        api.deploymentPlacements(deploymentId)
      ])
        .then(([progressValue, runValues, placementValues]) => {
          setProgress(progressValue);
          setRuns(runValues);
          setPlacements(placementValues);
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
          api.deploymentPlacements(deploymentId)
        ]).then(([runValues, placementValues]) => {
          setRuns(runValues);
          setPlacements(placementValues);
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
  const missingModelPath = runInputs.some((item) => !item.model_path);
  const isActive =
    progress?.operation &&
    ["running", "pausing", "cancelling"].includes(progress.operation.status);
  const canResume = progress?.operation?.status === "paused";

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
              The next editor checkpoint will expose deployment search dimensions here.
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
                      <td><code>{candidate.id}</code></td>
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
