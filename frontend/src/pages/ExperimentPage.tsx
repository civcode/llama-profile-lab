import { useEffect, useMemo, useState } from "react";
import { api, progressEvents } from "../api";
import {
  EmptyState,
  ErrorBanner,
  HeatmapTable,
  JsonDetails,
  MetricCard,
  PageHeader,
  ParetoPlot,
  StatusBadge
} from "../components";
import { workloadFilter, workloadLabel } from "../experiment";
import type {
  BinaryRecord,
  CandidateSummary,
  Experiment,
  ExperimentProgress,
  LauncherProfile,
  MatrixProjection,
  MetricDefinition,
  ParetoResult,
  Placement,
  RunSummary
} from "../types";

function numberMetric(metrics: Record<string, number>, name: string): string {
  const value = metrics[name];
  return value === undefined ? "—" : value.toFixed(1);
}

function bytes(value: number | undefined): string {
  if (value === undefined) return "—";
  const gib = value / 1024 / 1024 / 1024;
  return gib >= 1 ? gib.toFixed(2) + " GiB" : (value / 1024 / 1024).toFixed(0) + " MiB";
}

function profileForExperiment(
  experiment: Experiment,
  profiles: LauncherProfile[]
): LauncherProfile | null {
  const target = experiment.base_candidate.model.target_model_id;
  const prefix = "launcher-profile:";
  if (!target.startsWith(prefix)) return null;
  const id = target.slice(prefix.length);
  return profiles.find((item) => item.id === id) ?? null;
}

export function ExperimentPage({ experimentId }: { experimentId: string }) {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [progress, setProgress] = useState<ExperimentProgress | null>(null);
  const [candidates, setCandidates] = useState<CandidateSummary[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [binaries, setBinaries] = useState<BinaryRecord[]>([]);
  const [profiles, setProfiles] = useState<LauncherProfile[]>([]);
  const [placements, setPlacements] = useState<Placement[]>([]);
  const [metrics, setMetrics] = useState<MetricDefinition[]>([]);
  const [error, setError] = useState<unknown>(null);

  const [benchBinaryId, setBenchBinaryId] = useState("");
  const [fitBinaryId, setFitBinaryId] = useState("");
  const [modelPath, setModelPath] = useState("");
  const [telemetryInterval, setTelemetryInterval] = useState(1000);

  const [matrix, setMatrix] = useState<MatrixProjection | null>(null);
  const [matrixError, setMatrixError] = useState<unknown>(null);
  const [xPath, setXPath] = useState("");
  const [yPath, setYPath] = useState("");
  const [facetPath, setFacetPath] = useState("");
  const [metricName, setMetricName] = useState("throughput.median");
  const [workloadIndex, setWorkloadIndex] = useState(0);
  const [sliceValues, setSliceValues] = useState<Record<string, string>>({});

  const [pareto, setPareto] = useState<ParetoResult | null>(null);
  const [paretoError, setParetoError] = useState<unknown>(null);
  const [objectiveA, setObjectiveA] = useState(0);
  const [objectiveB, setObjectiveB] = useState(2);

  async function refresh() {
    try {
      const [
        experimentValue,
        progressValue,
        candidateValues,
        runValues,
        binaryValues,
        profileResponse,
        placementValues,
        metricValues
      ] = await Promise.all([
        api.experiment(experimentId),
        api.progress(experimentId),
        api.candidates(experimentId),
        api.runs(experimentId),
        api.binaries(),
        api.profiles(),
        api.placements(),
        api.metrics()
      ]);
      setExperiment(experimentValue);
      setProgress(progressValue);
      setCandidates(candidateValues);
      setRuns(runValues);
      setBinaries(binaryValues);
      setProfiles(profileResponse.items);
      setPlacements(placementValues);
      setMetrics(metricValues);

      const bench = binaryValues.find((item) => item.kind === "llama-bench");
      const fit = binaryValues.find((item) => item.kind === "llama-fit-params");
      if (bench && !benchBinaryId) setBenchBinaryId(bench.id);
      if (fit && !fitBinaryId) setFitBinaryId(fit.id);

      const sourceProfile = profileForExperiment(experimentValue, profileResponse.items);
      if (sourceProfile && !modelPath) setModelPath(sourceProfile.model_path);

      const dimensions = experimentValue.search_space.dimensions;
      if (!xPath && dimensions[0]) setXPath(dimensions[0].path);
      if (!yPath && dimensions[1]) setYPath(dimensions[1].path);
      else if (!yPath && dimensions[0]) setYPath(dimensions[0].path);
    } catch (reason) {
      setError(reason);
    }
  }

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => {
      void api.progress(experimentId).then(setProgress).catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(interval);
  }, [experimentId]);

  useEffect(() => {
    if (!progress?.operation) return;
    if (!["running", "pausing", "cancelling"].includes(progress.operation.status)) {
      return;
    }
    return progressEvents(
      experimentId,
      (next) => {
        setProgress(next);
        void Promise.all([api.runs(experimentId), api.candidates(experimentId)]).then(
          ([runValues, candidateValues]) => {
            setRuns(runValues);
            setCandidates(candidateValues);
          }
        );
      },
      () => void refresh()
    );
  }, [experimentId, progress?.operation?.id]);

  const dimensions = experiment?.search_space.dimensions ?? [];
  const remainingDimensions = dimensions.filter(
    (item) =>
      item.path !== xPath && item.path !== yPath && item.path !== facetPath
  );
  const profile = experiment ? profileForExperiment(experiment, profiles) : null;

  const matrixFilters = useMemo(() => {
    const filters = [workloadFilter(workloadIndex)];
    for (const dimension of remainingDimensions) {
      const value = sliceValues[dimension.path];
      if (value !== undefined && value !== "") {
        filters.push(dimension.path + "=" + value);
      }
    }
    return filters;
  }, [workloadIndex, remainingDimensions, sliceValues]);

  async function loadMatrix() {
    if (!experiment || !xPath || !yPath) return;
    try {
      setMatrixError(null);
      setMatrix(
        await api.matrix(experiment.id, {
          x: xPath,
          y: yPath,
          metric: metricName,
          facet: facetPath || null,
          filters: matrixFilters
        })
      );
    } catch (reason) {
      setMatrix(null);
      setMatrixError(reason);
    }
  }

  useEffect(() => {
    if (experiment && runs.some((run) => run.status === "completed")) {
      void loadMatrix();
    }
  }, [experiment?.id, xPath, yPath, facetPath, metricName, workloadIndex, runs.length]);

  async function execute(resume: boolean) {
    if (!experiment) return;
    try {
      setError(null);
      await api.run(
        experiment.id,
        {
          binary_id: benchBinaryId,
          fit_binary_id:
            experiment.definition.placement_policy.type === "fixed"
              ? null
              : fitBinaryId || null,
          model_path: modelPath,
          telemetry_interval_ms: telemetryInterval
        },
        resume
      );
      await refresh();
    } catch (reason) {
      setError(reason);
    }
  }

  async function loadPareto() {
    if (!experiment || experiment.workload_suite.cases.length < 2) return;
    try {
      setParetoError(null);
      setPareto(
        await api.pareto(experiment.id, [
          {
            key: workloadLabel(experiment.workload_suite.cases[objectiveA], objectiveA),
            direction: "maximize",
            metric: "throughput.median",
            filters: [{ path: "suite_case_index", value: objectiveA }]
          },
          {
            key: workloadLabel(experiment.workload_suite.cases[objectiveB], objectiveB),
            direction: "maximize",
            metric: "throughput.median",
            filters: [{ path: "suite_case_index", value: objectiveB }]
          }
        ])
      );
    } catch (reason) {
      setPareto(null);
      setParetoError(reason);
    }
  }

  if (!experiment || !progress) {
    return (
      <main className="page">
        <ErrorBanner error={error} />
        <div className="skeleton large" />
      </main>
    );
  }

  const latest = progress.latest_metrics;
  const failedCount = Object.entries(progress.case_status_counts)
    .filter(([status]) => !["completed", "planned", "running"].includes(status))
    .reduce((sum, [, value]) => sum + value, 0);
  const isActive =
    progress.operation &&
    ["running", "pausing", "cancelling"].includes(progress.operation.status);

  return (
    <main className="page">
      <PageHeader
        eyebrow={profile?.id ?? experiment.base_candidate.model.target_model_id}
        title={experiment.name}
        actions={
          <>
            <StatusBadge status={progress.experiment_status} />
            <a className="button" href="#/">
              All experiments
            </a>
          </>
        }
      />
      <ErrorBanner error={error} />

      <section className="metric-grid">
        <MetricCard
          label="Progress"
          value={progress.completed_cases + " / " + progress.total_cases}
          note={progress.incomplete_cases + " remaining"}
        />
        <MetricCard
          label="Latest throughput"
          value={
            progress.latest_tokens_per_second === null
              ? "—"
              : progress.latest_tokens_per_second.toFixed(1) + " t/s"
          }
          note={progress.latest_run_id ?? "No completed run yet"}
        />
        <MetricCard
          label="Process CPU"
          value={numberMetric(latest, "telemetry.process_cpu_avg_pct_normalized") + "%"}
          note={"System " + numberMetric(latest, "telemetry.cpu_system_avg_pct") + "%"}
        />
        <MetricCard
          label="GPU / VRAM"
          value={numberMetric(latest, "telemetry.gpu_utilization_avg_pct") + "%"}
          note={bytes(latest["telemetry.gpu_vram_used_peak_bytes"])}
        />
        <MetricCard
          label="Thermals"
          value={numberMetric(latest, "telemetry.gpu_temperature_peak_c") + "°C"}
          note={
            "CPU " +
            numberMetric(latest, "telemetry.cpu_temperature_peak_c") +
            "°C"
          }
        />
        <MetricCard
          label="Failures"
          value={failedCount}
          note="Persisted, never hidden"
        />
      </section>

      <section className="panel live-panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Execution</div>
              <h2>{isActive ? "Benchmarking in progress" : "Run controls"}</h2>
            </div>
            {progress.current_candidate_id ? (
              <div className="current-work">
                <span>Current candidate</span>
                <code>{progress.current_candidate_id}</code>
                <span>Workload</span>
                <code>{progress.current_workload_case_id}</code>
              </div>
            ) : null}
          </div>
          <div className="progress-track large">
            <span
              style={{
                width:
                  progress.total_cases > 0
                    ? Math.round(
                        (progress.completed_cases / progress.total_cases) * 100
                      ) + "%"
                    : "0%"
              }}
            />
          </div>
          <div className="run-form">
            <label className="field">
              <span>llama-bench</span>
              <select
                value={benchBinaryId}
                onChange={(event) => setBenchBinaryId(event.target.value)}
              >
                <option value="">Choose…</option>
                {binaries
                  .filter((item) => item.kind === "llama-bench")
                  .map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.path}
                    </option>
                  ))}
              </select>
            </label>
            {experiment.definition.placement_policy.type !== "fixed" ? (
              <label className="field">
                <span>llama-fit-params</span>
                <select
                  value={fitBinaryId}
                  onChange={(event) => setFitBinaryId(event.target.value)}
                >
                  <option value="">Choose…</option>
                  {binaries
                    .filter((item) => item.kind === "llama-fit-params")
                    .map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.path}
                      </option>
                    ))}
                </select>
              </label>
            ) : null}
            <label className="field grow">
              <span>Model path</span>
              <input
                value={modelPath}
                onChange={(event) => setModelPath(event.target.value)}
              />
            </label>
            <label className="field small-field">
              <span>Telemetry ms</span>
              <input
                type="number"
                min="500"
                value={telemetryInterval}
                onChange={(event) =>
                  setTelemetryInterval(Number(event.target.value))
                }
              />
            </label>
          </div>
          <div className="button-row">
            {!isActive && progress.experiment_status === "planned" ? (
              <button
                className="button button-primary"
                disabled={!benchBinaryId || !modelPath}
                onClick={() => void execute(false)}
              >
                Start experiment
              </button>
            ) : null}
            {!isActive &&
            ["paused", "failed"].includes(progress.experiment_status) ? (
              <button
                className="button button-primary"
                disabled={!benchBinaryId || !modelPath}
                onClick={() => void execute(true)}
              >
                Resume experiment
              </button>
            ) : null}
            {isActive ? (
              <>
                <button
                  className="button"
                  onClick={() => void api.pause(experiment.id).then(setProgress).catch(setError)}
                >
                  Pause
                </button>
                <button
                  className="button button-danger"
                  onClick={() => void api.cancel(experiment.id).then(setProgress).catch(setError)}
                >
                  Cancel
                </button>
              </>
            ) : null}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Observed points</div>
              <h2>Candidates</h2>
            </div>
            <span className="muted">{candidates.length} planned</span>
          </div>
          <div className="candidate-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  {dimensions.map((dimension) => (
                    <th key={dimension.path}>{dimension.path}</th>
                  ))}
                  <th>Runs</th>
                  <th>Server</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {candidates.map((item) => (
                  <tr key={item.id}>
                    <td>{item.ordinal + 1}</td>
                    {dimensions.map((dimension) => {
                      const path = dimension.path.split(".");
                      let current: unknown = item.candidate;
                      for (const segment of path) {
                        current =
                          current && typeof current === "object"
                            ? (current as Record<string, unknown>)[segment]
                            : undefined;
                      }
                      return <td key={dimension.path}>{String(current ?? "—")}</td>;
                    })}
                    <td>
                      {item.completed_case_count}/{item.benchmark_case_count}
                    </td>
                    <td>
                      {item.server_validation_count > 0
                        ? item.server_validation_count + " validated"
                        : "—"}
                    </td>
                    <td>
                      <a
                        className="text-link"
                        href={
                          "#/experiments/" +
                          experiment.id +
                          "/candidates/" +
                          item.id
                        }
                      >
                        Inspect
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Results</div>
              <h2>Matrix + heatmap</h2>
              <p className="section-copy">
                Pick two dimensions. Remaining dimensions become exact slices or an
                explicit facet; no hidden coordinate is averaged away.
              </p>
            </div>
            <button className="button" onClick={() => void loadMatrix()}>
              Refresh
            </button>
          </div>
          <div className="control-grid">
            <label className="field">
              <span>X dimension</span>
              <select value={xPath} onChange={(event) => setXPath(event.target.value)}>
                {dimensions.map((dimension) => (
                  <option value={dimension.path} key={dimension.path}>
                    {dimension.path}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Y dimension</span>
              <select value={yPath} onChange={(event) => setYPath(event.target.value)}>
                {dimensions.map((dimension) => (
                  <option value={dimension.path} key={dimension.path}>
                    {dimension.path}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Metric</span>
              <select
                value={metricName}
                onChange={(event) => setMetricName(event.target.value)}
              >
                {metrics.map((item) => (
                  <option value={item.name} key={item.name}>
                    {item.label} {item.unit ? "(" + item.unit + ")" : ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Workload</span>
              <select
                value={workloadIndex}
                onChange={(event) => setWorkloadIndex(Number(event.target.value))}
              >
                {experiment.workload_suite.cases.map((item, index) => (
                  <option key={index} value={index}>
                    {workloadLabel(item, index)}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Facet</span>
              <select
                value={facetPath}
                onChange={(event) => setFacetPath(event.target.value)}
              >
                <option value="">None</option>
                {dimensions
                  .filter((item) => item.path !== xPath && item.path !== yPath)
                  .map((item) => (
                    <option value={item.path} key={item.path}>
                      {item.path}
                    </option>
                  ))}
              </select>
            </label>
            {remainingDimensions.map((dimension) => (
              <label className="field" key={dimension.path}>
                <span>Slice {dimension.path}</span>
                <select
                  value={
                    sliceValues[dimension.path] ??
                    String(dimension.values[0] ?? "")
                  }
                  onChange={(event) =>
                    setSliceValues((current) => ({
                      ...current,
                      [dimension.path]: event.target.value
                    }))
                  }
                >
                  {dimension.values.map((value) => (
                    <option value={String(value)} key={String(value)}>
                      {String(value)}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          <ErrorBanner error={matrixError} />
          {matrix ? (
            <HeatmapTable matrix={matrix} />
          ) : (
            <EmptyState title="No matrix yet.">
              Complete benchmark runs, then choose a workload slice.
            </EmptyState>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Trade-offs</div>
              <h2>Pareto frontier</h2>
              <p className="section-copy">
                Choose two workload objectives. The plot shows only non-dominated
                observed Candidates; it does not collapse them into a universal score.
              </p>
            </div>
            <button className="button" onClick={() => void loadPareto()}>
              Calculate
            </button>
          </div>
          <div className="control-grid two">
            <label className="field">
              <span>Objective X</span>
              <select
                value={objectiveA}
                onChange={(event) => setObjectiveA(Number(event.target.value))}
              >
                {experiment.workload_suite.cases.map((item, index) => (
                  <option value={index} key={index}>
                    {workloadLabel(item, index)}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Objective Y</span>
              <select
                value={objectiveB}
                onChange={(event) => setObjectiveB(Number(event.target.value))}
              >
                {experiment.workload_suite.cases.map((item, index) => (
                  <option value={index} key={index}>
                    {workloadLabel(item, index)}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <ErrorBanner error={paretoError} />
          {pareto ? <ParetoPlot result={pareto} /> : null}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="eyebrow">Recent attempts</div>
          <h2>Runs</h2>
          <div className="candidate-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Started</th>
                  <th>Candidate</th>
                  <th>Workload</th>
                  <th>Status</th>
                  <th>Quality</th>
                </tr>
              </thead>
              <tbody>
                {runs
                  .slice()
                  .reverse()
                  .slice(0, 12)
                  .map((run) => (
                    <tr key={run.id}>
                      <td>{new Date(run.started_at).toLocaleTimeString()}</td>
                      <td><code>{run.candidate_id}</code></td>
                      <td>{run.workload_kind}</td>
                      <td><StatusBadge status={run.status} /></td>
                      <td>{run.quality ?? "—"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <JsonDetails label="Experiment definition" value={experiment} />
          {experiment.definition.placement_policy.type === "fixed" ? (
            <JsonDetails
              label="Fixed placement"
              value={placements.find(
                (item) =>
                  item.id === experiment.definition.placement_policy.placement_id
              )}
            />
          ) : null}
        </div>
      </section>
    </main>
  );
}
