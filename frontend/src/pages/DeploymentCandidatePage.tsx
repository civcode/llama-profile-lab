import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
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
  DeploymentPlacement,
  DeploymentPromotionResponse,
  DeploymentRun,
  Experiment
} from "../types";

function numeric(row: Record<string, unknown>, key: string): number | null {
  const value = row[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function text(row: Record<string, unknown>, key: string): string | null {
  const value = row[key];
  return typeof value === "string" ? value : null;
}

function metric(value: number | null, suffix = ""): string {
  return value === null ? "—" : value.toFixed(2) + suffix;
}

export function DeploymentCandidatePage({
  deploymentId,
  candidateId
}: {
  deploymentId: string;
  candidateId: string;
}) {
  const [candidate, setCandidate] = useState<DeploymentCandidateItem | null>(null);
  const [deployment, setDeployment] = useState<Deployment | null>(null);
  const [placements, setPlacements] = useState<DeploymentPlacement[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [sourceOptions, setSourceOptions] = useState<
    Record<string, { id: string; name: string }[]>
  >({});
  const [sourceExperiments, setSourceExperiments] = useState<Record<string, string>>({});
  const [promotionPlacementId, setPromotionPlacementId] = useState("");
  const [promotion, setPromotion] = useState<DeploymentPromotionResponse | null>(null);
  const [promotionError, setPromotionError] = useState<unknown>(null);
  const [promoting, setPromoting] = useState(false);
  const [runs, setRuns] = useState<DeploymentRun[]>([]);
  const [results, setResults] = useState<Record<string, unknown>[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    Promise.all([
      api.deploymentCandidates(deploymentId),
      api.deploymentPlacements(deploymentId),
      api.deploymentRuns(deploymentId),
      api.deploymentResults(deploymentId, [
        "deployment.candidate_id=" + candidateId
      ]),
      api.deployment(deploymentId),
      api.experiments()
    ])
      .then(async ([
        candidateValues,
        placementValues,
        runValues,
        resultValues,
        deploymentValue,
        experimentValues
      ]) => {
        const experimentCandidates = await Promise.all(
          experimentValues.map(async (experiment) => ({
            experiment,
            candidates: await api.candidates(experiment.id)
          }))
        );
        if (!alive) return;
        const selected = candidateValues.find((item) => item.id === candidateId) ?? null;
        setCandidate(selected);
        setDeployment(deploymentValue);
        setExperiments(experimentValues);
        const candidatePlacements = placementValues.filter(
          (item) => item.deployment_candidate_id === candidateId
        );
        setPlacements(candidatePlacements);
        setPromotionPlacementId((current) => current || candidatePlacements[0]?.id || "");
        const nextOptions: Record<string, { id: string; name: string }[]> = {};
        const defaults: Record<string, string> = {};
        for (const instance of deploymentValue.definition.instances) {
          const matches = experimentCandidates
            .filter(({ candidates }) =>
              candidates.some((item) => item.id === instance.candidate_id)
            )
            .map(({ experiment }) => ({
              id: experiment.id,
              name: experiment.name
            }));
          nextOptions[instance.instance_id] = matches;
          if (matches[0]) defaults[instance.instance_id] = matches[0].id;
        }
        setSourceOptions(nextOptions);
        setSourceExperiments((current) => ({ ...defaults, ...current }));
        setRuns(
          runValues.filter((item) => item.deployment_candidate_id === candidateId)
        );
        setResults(resultValues);
        if (!selected) {
          setError(new Error("Deployment Candidate not found in this plan."));
        }
      })
      .catch((reason) => alive && setError(reason))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [deploymentId, candidateId]);

  const instanceIds = candidate?.definition.instances.map((item) => item.instance_id) ?? [];
  const sourceCoverageComplete = useMemo(() => {
    if (!deployment) return false;
    return deployment.definition.instances.every(
      (instance) => Boolean(sourceExperiments[instance.instance_id])
    );
  }, [deployment, sourceExperiments]);

  const validResults = useMemo(
    () =>
      results.filter(
        (row) =>
          row.correctness_valid === true &&
          row.workload_status === "completed" &&
          row.deployment_status === "completed"
      ),
    [results]
  );

  async function proposePromotion() {
    if (!deployment || !promotionPlacementId || !sourceCoverageComplete) return;
    try {
      setPromoting(true);
      setPromotionError(null);
      setPromotion(
        await api.promoteDeployment(deploymentId, {
          deployment_placement_id: promotionPlacementId,
          sources: deployment.definition.instances.map((instance) => ({
            instance_id: instance.instance_id,
            experiment_id: sourceExperiments[instance.instance_id]
          }))
        })
      );
    } catch (reason) {
      setPromotion(null);
      setPromotionError(reason);
    } finally {
      setPromoting(false);
    }
  }

  if (loading || !candidate) {
    return (
      <main className="page">
        <ErrorBanner error={error} />
        {loading ? <div className="skeleton large" /> : null}
      </main>
    );
  }

  const latestResult =
    validResults.length === 0 ? null : validResults[validResults.length - 1];
  const latestRun = runs.length === 0 ? null : runs[runs.length - 1];

  return (
    <main className="page">
      <PageHeader
        eyebrow="Deployment Candidate"
        title={candidate.definition.instances
          .map((item) => item.role || item.instance_id)
          .join(" + ")}
        actions={
          <>
            {latestRun ? <StatusBadge status={latestRun.status} /> : null}
            <a className="button" href={"#/deployments/" + deploymentId}>
              Back to deployment
            </a>
          </>
        }
      />
      <ErrorBanner error={error} />

      <section className="metric-grid">
        <MetricCard
          label="Placements"
          value={candidate.placement_ids.length}
          note={candidate.id}
        />
        <MetricCard
          label="Rejections"
          value={candidate.rejection_count}
          note="Persisted pruning evidence"
        />
        <MetricCard
          label="DD decode"
          value={metric(
            latestResult ? numeric(latestResult, "combined_tg_tps") : null,
            " t/s"
          )}
          note="Latest valid observation"
        />
        <MetricCard
          label="PP prefill"
          value={metric(
            latestResult ? numeric(latestResult, "combined_pp_tps") : null,
            " t/s"
          )}
          note="Latest valid observation"
        />
        <MetricCard
          label="Minimum retention"
          value={metric(
            latestResult === null ||
              numeric(latestResult, "min_retention") === null
              ? null
              : numeric(latestResult, "min_retention")! * 100,
            "%"
          )}
          note="Across concurrent members"
        />
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Search coordinates</div>
              <h2>How this Candidate was generated</h2>
            </div>
          </div>
          <JsonDetails label="Generation metadata" value={candidate.generation} />
          <JsonDetails label="Deployment definition" value={candidate.definition} />
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Pruning</div>
              <h2>Rejection explanations</h2>
              <p className="section-copy">
                Capability, estimator, and memory failures remain queryable even when
                the Candidate has no feasible placement.
              </p>
            </div>
          </div>
          {candidate.rejections.length === 0 ? (
            <EmptyState title="No persisted rejection events for this Candidate." />
          ) : (
            <div className="rejection-list">
              {candidate.rejections.map((rejection) => (
                <article className="rejection-card" key={rejection.id}>
                  <div className="card-topline">
                    <strong>{rejection.reason.replaceAll("_", " ")}</strong>
                    <span className="status">{rejection.stage}</span>
                  </div>
                  <div className="muted">
                    {new Date(rejection.created_at).toLocaleString()}
                  </div>
                  <JsonDetails label="Rejection details" value={rejection.details} />
                </article>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Interference</div>
              <h2>Standalone versus concurrent behavior</h2>
              <p className="section-copy">
                Each row preserves the standalone denominator, overlap throughput,
                retention, and latency change for the exact DD/PP/PD/DP workload.
              </p>
            </div>
          </div>
          {results.length === 0 ? (
            <EmptyState title="No concurrent workload observations yet." />
          ) : (
            <div className="interference-stack">
              {results.map((row, index) => {
                const phase = text(row, "phase") ?? "—";
                const placementId = text(row, "deployment_placement_id") ?? "—";
                const valid = row.correctness_valid === true;
                return (
                  <article
                    className={"interference-card " + (valid ? "" : "is-invalid")}
                    key={(text(row, "workload_run_id") ?? placementId) + ":" + index}
                  >
                    <div className="section-heading-row">
                      <div>
                        <strong>{phase.toUpperCase()} phase</strong>
                        <div className="muted"><code>{placementId}</code></div>
                      </div>
                      <StatusBadge
                        status={
                          valid
                            ? String(row.workload_status ?? "completed")
                            : "invalid"
                        }
                      />
                    </div>
                    <div className="metric-grid compact">
                      <MetricCard
                        label="Combined PP"
                        value={metric(numeric(row, "combined_pp_tps"), " t/s")}
                      />
                      <MetricCard
                        label="Combined TG"
                        value={metric(numeric(row, "combined_tg_tps"), " t/s")}
                      />
                      <MetricCard
                        label="Min retention"
                        value={metric(
                          numeric(row, "min_retention") === null
                            ? null
                            : numeric(row, "min_retention")! * 100,
                          "%"
                        )}
                      />
                    </div>
                    <div className="candidate-table-wrap">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Instance</th>
                            <th>Mode</th>
                            <th>Standalone TPS</th>
                            <th>Concurrent TPS</th>
                            <th>Retention</th>
                            <th>Latency</th>
                            <th>Latency Δ</th>
                          </tr>
                        </thead>
                        <tbody>
                          {instanceIds.map((instanceId) => {
                            const prefix = "instance." + instanceId + ".";
                            const retention = numeric(row, prefix + "retention");
                            return (
                              <tr key={instanceId}>
                                <td><strong>{instanceId}</strong></td>
                                <td>{text(row, prefix + "mode") ?? "—"}</td>
                                <td>{metric(numeric(row, prefix + "standalone_tps"))}</td>
                                <td>
                                  {metric(
                                    numeric(row, prefix + "overlap_tps") ??
                                      numeric(row, prefix + "native_tps")
                                  )}
                                </td>
                                <td>
                                  {metric(
                                    retention === null ? null : retention * 100,
                                    "%"
                                  )}
                                </td>
                                <td>{metric(numeric(row, prefix + "latency_ms"), " ms")}</td>
                                <td>
                                  {metric(
                                    numeric(row, prefix + "latency_increase_pct"),
                                    "%"
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Placements</div>
              <h2>Memory feasibility</h2>
            </div>
          </div>
          {placements.length === 0 ? (
            <EmptyState title="No feasible placements for this Candidate." />
          ) : (
            <div className="candidate-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Placement</th>
                    <th>Feasibility</th>
                    {placements[0]?.memory.devices.map((device) => (
                      <th key={device}>{device} projected free</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {placements.map((placement) => {
                    const projected = placement.memory.rows.find(
                      (row) => row.key === "projected_free"
                    );
                    return (
                      <tr key={placement.id}>
                        <td><code>{placement.id}</code></td>
                        <td><StatusBadge status={placement.feasibility} /></td>
                        {placement.memory.devices.map((device) => (
                          <td key={device}>
                            {metric(
                              projected?.values[device] === null ||
                                projected?.values[device] === undefined
                                ? null
                                : projected.values[device] / 1024 / 1024 / 1024,
                              " GiB"
                            )}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <section className="panel promotion-panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Promotion</div>
              <h2>Coordinated launcher proposal</h2>
              <p className="section-copy">
                Generate one review-only patch for every model profile participating
                in this finalist. The proposal is refused if any source profile,
                estimator provenance, placement evidence, or configured concurrency
                phase is missing.
              </p>
            </div>
          </div>

          <div className="control-grid two">
            <label className="field">
              <span>Finalist placement</span>
              <select
                value={promotionPlacementId}
                onChange={(event) => setPromotionPlacementId(event.target.value)}
              >
                <option value="">Choose…</option>
                {placements.map((placement) => (
                  <option key={placement.id} value={placement.id}>
                    {placement.id}
                  </option>
                ))}
              </select>
            </label>
            <div className="promotion-source-grid">
              {deployment?.definition.instances.map((instance) => (
                <label className="field" key={instance.instance_id}>
                  <span>{instance.instance_id} source experiment</span>
                  <select
                    value={sourceExperiments[instance.instance_id] ?? ""}
                    onChange={(event) =>
                      setSourceExperiments((current) => ({
                        ...current,
                        [instance.instance_id]: event.target.value
                      }))
                    }
                  >
                    <option value="">Choose…</option>
                    {(sourceOptions[instance.instance_id] ?? []).map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.name}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          </div>
          {!sourceCoverageComplete ? (
            <div className="banner banner-error">
              Every base deployment instance must resolve to a source experiment
              before a coordinated proposal can be generated.
            </div>
          ) : null}
          <ErrorBanner error={promotionError} />
          <div className="button-row">
            <button
              className="button button-primary"
              disabled={
                promoting ||
                !promotionPlacementId ||
                !sourceCoverageComplete
              }
              onClick={() => void proposePromotion()}
            >
              {promoting ? "Generating…" : "Generate coordinated proposal"}
            </button>
          </div>

          {promotion ? (
            <div className="promotion-review">
              <div className="metric-grid compact">
                <MetricCard
                  label="Profiles"
                  value={promotion.sources.length}
                  note="All instances included"
                />
                <MetricCard
                  label="Placement"
                  value={<code>{promotion.deployment_placement_id}</code>}
                  note="Frozen finalist topology"
                />
                <MetricCard
                  label="Proposal"
                  value={<code>{promotion.id}</code>}
                  note="Append-only review record"
                />
              </div>
              <div className="promotion-change-list">
                {promotion.changes.map((item) => (
                  <div className="promotion-change-card" key={item.instance_id}>
                    <strong>
                      {item.instance_id} · {item.source_profile_id}
                    </strong>
                    <div className="muted">
                      {item.changes.length} launcher argument change
                      {item.changes.length === 1 ? "" : "s"}
                    </div>
                    <JsonDetails label="Argument changes" value={item.changes} />
                  </div>
                ))}
              </div>
              <div className="field-label">Unified launcher patch</div>
              <pre className="promotion-patch">{promotion.patch || "No textual changes."}</pre>
              <JsonDetails label="Promotion evidence" value={promotion.evidence} />
            </div>
          ) : null}
        </div>
      </section>
    </main>
  );
}
