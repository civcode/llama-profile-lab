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
  const [objectiveAMetric, setObjectiveAMetric] = useState("throughput.median");
  const [objectiveBMetric, setObjectiveBMetric] = useState("throughput.median");
  const [objectiveADirection, setObjectiveADirection] = useState<
    "maximize" | "minimize"
  >("maximize");
  const [objectiveBDirection, setObjectiveBDirection] = useState<
    "maximize" | "minimize"
  >("maximize");

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

      const workloadCount = experimentValue.workload_suite.cases.length;
      if (workloadCount > 0) {
        setObjectiveA((current) => Math.min(current, workloadCount - 1));
        setObjectiveB((current) => Math.min(current, workloadCount - 1));
      }
    } catch (reason) {
      setError(reason);
    }
  }

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => {
      void Promise.all([
        api.progress(experimentId),
        api.runs(experimentId),
        api.candidates(experimentId)
      ])
        .then(([progressValue, runValues, candidateValues]) => {
          setProgress(progressValue);
          setRuns(runValues);
          setCandidates(candidateValues);
        })
        .catch(() => undefined);
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
      const objectiveALabel = workloadLabel(
        experiment.workload_suite.cases[objectiveA],
        objectiveA
      );
      const objectiveBLabel = workloadLabel(
        experiment.workload_suite.cases[objectiveB],
        objectiveB
      );
      setPareto(
        await api.pareto(experiment.id, [
          {
            key: "X · " + objectiveALabel + " · " + objectiveAMetric,
            direction: objectiveADirection,
            metric: objectiveAMetric,
            filters: [{ path: "suite_case_index", value: objectiveA }]
          },
          {
            key: "Y · " + objectiveBLabel + " · " + objectiveBMetric,
            direction: objectiveBDirection,
            metric: objectiveBMetric,
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
  const fixedPlacementId =
    experiment.definition.placement_policy.type === "fixed"
      ? experiment.definition.placement_policy.placement_id
      : null;
  const fixedPlacement =
    fixedPlacementId === null
      ? null
      : placements.find((item) => item.id === fixedPlacementId) ?? null;

  return (
    <main className="page">
      <PageHeader
        eyebrow={profile?.id ?? experiment.base_candidate.model.target_model_id}
        title={experiment.name}
        actions={
          <>
            <StatusBadge status={progress.experiment_status} />
            <a className="button" href={"#/experiments/" + experiment.id + "/compare"}>
              Compare Candidates
            </a>
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
          note={progress.latest_run_id ? "Latest completed run" : "No completed run yet"}
        />
        <MetricCard
          label="Process CPU"
          value={numberMetric(latest, "telemetry.process_cpu_avg_pct_normalized") + "%"}
          note={"System " + numberMetric(latest, "telemetry.cpu_system_avg_pct") + "%"}
        />
        <MetricCard
          label="GPU utilization"
          value={numberMetric(latest, "telemetry.gpu_utilization_avg_pct") + "%"}
          note={
            "Power " +
            numberMetric(latest, "telemetry.gpu_power_avg_w") +
            " W"
          }
        />
        <MetricCard
          label="Memory"
          value={bytes(latest["telemetry.gpu_vram_used_peak_bytes"])}
          note={
            "RAM " +
            bytes(latest["telemetry.ram_used_peak_bytes"])
          }
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
                <strong>
                  Candidate #
                  {(progress.current_candidate_ordinal ?? 0) + 1}
                </strong>
                <span>Workload</span>
                <strong>
                  {progress.current_suite_case_index === null
                    ? "Running"
                    : workloadLabel(
                        experiment.workload_suite.cases[
                          progress.current_suite_case_index
                        ],
                        progress.current_suite_case_index
                      )}
                </strong>
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
          <div className="control-grid three">
            <label className="field">
              <span>Objective X workload</span>
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
              <span>Objective X metric</span>
              <select
                value={objectiveAMetric}
                onChange={(event) => setObjectiveAMetric(event.target.value)}
              >
                {metrics.map((item) => (
                  <option value={item.name} key={item.name}>
                    {item.label} {item.unit ? "(" + item.unit + ")" : ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Objective X direction</span>
              <select
                value={objectiveADirection}
                onChange={(event) =>
                  setObjectiveADirection(
                    event.target.value as "maximize" | "minimize"
                  )
                }
              >
                <option value="maximize">Maximize</option>
                <option value="minimize">Minimize</option>
              </select>
            </label>
            <label className="field">
              <span>Objective Y workload</span>
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
            <label className="field">
              <span>Objective Y metric</span>
              <select
                value={objectiveBMetric}
                onChange={(event) => setObjectiveBMetric(event.target.value)}
              >
                {metrics.map((item) => (
                  <option value={item.name} key={item.name}>
                    {item.label} {item.unit ? "(" + item.unit + ")" : ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Objective Y direction</span>
              <select
                value={objectiveBDirection}
                onChange={(event) =>
                  setObjectiveBDirection(
                    event.target.value as "maximize" | "minimize"
                  )
                }
              >
                <option value="maximize">Maximize</option>
                <option value="minimize">Minimize</option>
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
                      <td>
                        Candidate #
                        {(candidates.find((item) => item.id === run.candidate_id)
                          ?.ordinal ?? 0) + 1}
                      </td>
                      <td>{run.workload_kind}</td>
                      <td><StatusBadge status={run.status} /></td>
                      <td>{run.quality ?? "—"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <JsonDetails label="Experiment definition" value={experiment} />
          {fixedPlacementId !== null ? (
            <JsonDetails label="Fixed placement" value={fixedPlacement} />
          ) : null}
        </div>
      </section>
    </main>
  );
}
