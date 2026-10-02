import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import {
  DepthCurve,
  EmptyState,
  ErrorBanner,
  JsonDetails,
  MetricCard,
  PageHeader,
  StatusBadge
} from "../components";
import { workloadLabel } from "../experiment";
import type {
  BinaryRecord,
  CandidateComparison,
  CandidateSummary,
  CandidateValidationHistory,
  Experiment,
  LatencyEstimate,
  LauncherProfile,
  Placement,
  PromotionResponse,
  ResultRow
} from "../types";

const comparisonMetrics = [
  "throughput.median",
  "throughput.cv",
  "cpu.process.avg_pct",
  "cpu.system.avg_pct",
  "gpu.utilization.avg_pct",
  "gpu.vram.peak_bytes",
  "memory.process_rss.peak_bytes"
];

function sourceProfile(
  experiment: Experiment,
  profiles: LauncherProfile[]
): LauncherProfile | null {
  const prefix = "launcher-profile:";
  const target = experiment.base_candidate.model.target_model_id;
  if (!target.startsWith(prefix)) return null;
  return profiles.find((item) => item.id === target.slice(prefix.length)) ?? null;
}

function formatMetric(value: unknown): string {
  if (typeof value !== "number") return "—";
  return Math.abs(value) >= 1000 ? value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : value.toFixed(2);
}

export function CandidatePage({
  experimentId,
  candidateId
}: {
  experimentId: string;
  candidateId: string;
}) {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [candidate, setCandidate] = useState<CandidateSummary | null>(null);
  const [results, setResults] = useState<ResultRow[]>([]);
  const [comparison, setComparison] = useState<CandidateComparison | null>(null);
  const [placements, setPlacements] = useState<Placement[]>([]);
  const [history, setHistory] = useState<CandidateValidationHistory | null>(null);
  const [profiles, setProfiles] = useState<LauncherProfile[]>([]);
  const [binaries, setBinaries] = useState<BinaryRecord[]>([]);
  const [error, setError] = useState<unknown>(null);

  const [latency, setLatency] = useState<LatencyEstimate | null>(null);
  const [latencyError, setLatencyError] = useState<unknown>(null);
  const [promptTokens, setPromptTokens] = useState(2048);
  const [generateTokens, setGenerateTokens] = useState(256);
  const [decodeDepth, setDecodeDepth] = useState(4096);

  const [serverBinaryId, setServerBinaryId] = useState("");
  const [speedBinaryId, setSpeedBinaryId] = useState("");
  const [validationPort, setValidationPort] = useState(8080);
  const [validating, setValidating] = useState(false);
  const [promotion, setPromotion] = useState<PromotionResponse | null>(null);
  const [promoting, setPromoting] = useState(false);

  async function refresh() {
    try {
      const [
        experimentValue,
        candidates,
        resultRows,
        placementItems,
        validationHistory,
        profileResponse,
        binaryItems
      ] = await Promise.all([
        api.experiment(experimentId),
        api.candidates(experimentId),
        api.results(experimentId, ["run.candidate_id=" + candidateId], comparisonMetrics),
        api.placements(),
        api.validationHistory(experimentId, candidateId),
        api.profiles(),
        api.binaries()
      ]);
      setExperiment(experimentValue);
      setCandidate(candidates.find((item) => item.id === candidateId) ?? null);
      setResults(resultRows);
      setPlacements(placementItems.filter((item) => item.candidate_id === candidateId));
      setHistory(validationHistory);
      setProfiles(profileResponse.items);
      setBinaries(binaryItems);
      const server = binaryItems.find((item) => item.kind === "llama-server");
      const speed = binaryItems.find((item) => item.kind === "speed-bench");
      if (server && !serverBinaryId) setServerBinaryId(server.id);
      if (speed && !speedBinaryId) setSpeedBinaryId(speed.id);

      if (candidates.some((item) => item.id === candidateId) && resultRows.length > 0) {
        try {
          setComparison(
            await api.compare(
              experimentId,
              candidateId,
              comparisonMetrics
            )
          );
        } catch {
          setComparison(null);
        }
      }
    } catch (reason) {
      setError(reason);
    }
  }

  useEffect(() => {
    void refresh();
  }, [experimentId, candidateId]);

  const profile = experiment ? sourceProfile(experiment, profiles) : null;
  const rowsByWorkload = useMemo(
    () => results.slice().sort((a, b) => a.suite_case_index - b.suite_case_index),
    [results]
  );

  async function estimateLatency() {
    try {
      setLatencyError(null);
      setLatency(
        await api.latency(
          experimentId,
          candidateId,
          promptTokens,
          generateTokens,
          decodeDepth
        )
      );
    } catch (reason) {
      setLatency(null);
      setLatencyError(reason);
    }
  }

  async function validate() {
    if (!experiment || !candidate || !profile) return;
    try {
      setValidating(true);
      setError(null);
      const placement = placements[placements.length - 1] ?? null;
      await api.validateCandidate(candidate.id, {
        experiment_id: experiment.id,
        server_binary_id: serverBinaryId,
        speed_bench_binary_id: speedBinaryId,
        model_path: profile.model_path,
        draft_model_path: profile.draft_model_path,
        placement_id: placement?.id ?? null,
        model_name: profile.server_alias ?? profile.id,
        host: "127.0.0.1",
        port: validationPort,
        readiness_timeout_seconds: 300,
        request_timeout_seconds: 600,
        benchmark_timeout_seconds: null,
        workload_case_id: null
      });
      await refresh();
    } catch (reason) {
      setError(reason);
    } finally {
      setValidating(false);
    }
  }

  async function promote() {
    if (!experiment || !candidate || !profile) return;
    try {
      setPromoting(true);
      setError(null);
      setPromotion(
        await api.promoteCandidate(candidate.id, {
          experiment_id: experiment.id,
          source_profile_id: profile.id
        })
      );
      await refresh();
    } catch (reason) {
      setError(reason);
    } finally {
      setPromoting(false);
    }
  }

  if (!experiment || !candidate) {
    return (
      <main className="page">
        <ErrorBanner error={error} />
        <div className="skeleton large" />
      </main>
    );
  }

  const throughput = rowsByWorkload.map((row) => row["throughput.median"]).find(
    (value): value is number => typeof value === "number"
  );
  const processCpu = rowsByWorkload.map((row) => row["cpu.process.avg_pct"]).find(
    (value): value is number => typeof value === "number"
  );
  const gpu = rowsByWorkload.map((row) => row["gpu.utilization.avg_pct"]).find(
    (value): value is number => typeof value === "number"
  );
  const stability = rowsByWorkload.map((row) => row["throughput.cv"]).find(
    (value): value is number => typeof value === "number"
  );
  const decodeCurve = rowsByWorkload
    .filter(
      (row) =>
        row.workload_kind === "microbench-decode" &&
        typeof row.depth_tokens === "number" &&
        typeof row["throughput.median"] === "number"
    )
    .map((row) => ({
      depth: row.depth_tokens as number,
      throughput: row["throughput.median"] as number,
      label: workloadLabel(
        experiment.workload_suite.cases[row.suite_case_index],
        row.suite_case_index
      )
    }));

  return (
    <main className="page">
      <PageHeader
        eyebrow={"Candidate #" + (candidate.ordinal + 1)}
        title={experiment.name}
        actions={
          <a className="button" href={"#/experiments/" + experiment.id}>
            Back to experiment
          </a>
        }
      />
      <ErrorBanner error={error} />

      <section className="metric-grid">
        <MetricCard
          label="Representative throughput"
          value={throughput === undefined ? "—" : throughput.toFixed(1) + " t/s"}
        />
        <MetricCard
          label="Process CPU"
          value={processCpu === undefined ? "—" : processCpu.toFixed(1) + "%"}
        />
        <MetricCard
          label="GPU utilization"
          value={gpu === undefined ? "—" : gpu.toFixed(1) + "%"}
        />
        <MetricCard
          label="Throughput CV"
          value={stability === undefined ? "—" : (stability * 100).toFixed(2) + "%"}
        />
        <MetricCard
          label="Server validation"
          value={
            candidate.server_validation_count > 0
              ? candidate.server_validation_count + " complete"
              : "Not validated"
          }
        />
      </section>

      <div className="two-column">
        <section className="panel">
          <div className="section-body">
            <div className="eyebrow">Configuration</div>
            <h2>Candidate</h2>
            <div className="definition-grid">
              <span>Context</span>
              <strong>{candidate.candidate.context.size.toLocaleString()}</strong>
              <span>Batch / ubatch</span>
              <strong>
                {candidate.candidate.compute.batch_size} /{" "}
                {candidate.candidate.compute.ubatch_size}
              </strong>
              <span>KV cache</span>
              <strong>
                {candidate.candidate.context.cache_type_k} /{" "}
                {candidate.candidate.context.cache_type_v}
              </strong>
              <span>Flash attention</span>
              <strong>{candidate.candidate.compute.flash_attn}</strong>
            </div>
            <JsonDetails value={candidate.candidate} />
          </div>
        </section>

        <section className="panel">
          <div className="section-body">
            <div className="eyebrow">Resolved placement</div>
            <h2>{placements.length ? "Frozen host placement" : "Not resolved yet"}</h2>
            {placements.length ? (
              <div className="definition-grid">
                <span>GPU layers</span>
                <strong>{placements[placements.length - 1].n_gpu_layers}</strong>
                <span>CPU MoE</span>
                <strong>{placements[placements.length - 1].n_cpu_moe}</strong>
                <span>Split</span>
                <strong>{placements[placements.length - 1].split_mode}</strong>
                <span>Production context</span>
                <strong>
                  {placements[
                    placements.length - 1
                  ].production_context_size.toLocaleString()}
                </strong>
              </div>
            ) : (
              <p className="section-copy">
                Placement appears after fitting or after a fixed placement is assigned.
              </p>
            )}
            {placements.length ? (
              <JsonDetails value={placements[placements.length - 1]} />
            ) : null}
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="section-body">
          <div className="eyebrow">Performance + resources</div>
          <h2>Workload observations</h2>
          {rowsByWorkload.length ? (
            <div className="candidate-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Workload</th>
                    <th>Median t/s</th>
                    <th>CV</th>
                    <th>Process CPU</th>
                    <th>System CPU</th>
                    <th>GPU</th>
                    <th>VRAM</th>
                    <th>RSS</th>
                  </tr>
                </thead>
                <tbody>
                  {rowsByWorkload.map((row) => (
                    <tr key={row.workload_case_id}>
                      <td>
                        {workloadLabel(
                          experiment.workload_suite.cases[row.suite_case_index],
                          row.suite_case_index
                        )}
                      </td>
                      <td>{formatMetric(row["throughput.median"])}</td>
                      <td>{formatMetric(row["throughput.cv"])}</td>
                      <td>{formatMetric(row["cpu.process.avg_pct"])}</td>
                      <td>{formatMetric(row["cpu.system.avg_pct"])}</td>
                      <td>{formatMetric(row["gpu.utilization.avg_pct"])}</td>
                      <td>{formatMetric(row["gpu.vram.peak_bytes"])}</td>
                      <td>{formatMetric(row["memory.process_rss.peak_bytes"])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState title="No completed microbenchmarks for this Candidate." />
          )}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="eyebrow">Decode scaling</div>
          <h2>Throughput by active context depth</h2>
          <p className="section-copy">
            The curve uses only measured decode workloads for this Candidate; no
            synthetic depth points are inserted.
          </p>
          <DepthCurve points={decodeCurve} />
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="eyebrow">Baseline</div>
          <h2>Signed deltas</h2>
          {comparison ? (
            <div className="candidate-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Workload</th>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Candidate</th>
                    <th>Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.deltas.map((delta, index) => (
                    <tr key={index}>
                      <td>{delta.workload_label}</td>
                      <td>{delta.metric}</td>
                      <td>{formatMetric(delta.baseline_value)}</td>
                      <td>{formatMetric(delta.candidate_value)}</td>
                      <td>
                        {delta.percent_delta === null
                          ? "—"
                          : (delta.percent_delta >= 0 ? "+" : "") +
                            delta.percent_delta.toFixed(2) +
                            "%"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState title="Baseline comparison becomes available after matching runs complete." />
          )}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Request simulation</div>
              <h2>Compute-only latency</h2>
            </div>
            <button className="button" onClick={() => void estimateLatency()}>
              Estimate
            </button>
          </div>
          <div className="control-grid three">
            <label className="field">
              <span>Prompt tokens</span>
              <input
                type="number"
                min="1"
                value={promptTokens}
                onChange={(event) => setPromptTokens(Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Generate tokens</span>
              <input
                type="number"
                min="1"
                value={generateTokens}
                onChange={(event) => setGenerateTokens(Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Decode start depth</span>
              <input
                type="number"
                min="0"
                value={decodeDepth}
                onChange={(event) => setDecodeDepth(Number(event.target.value))}
              />
            </label>
          </div>
          <ErrorBanner error={latencyError} />
          {latency ? (
            <div className="metric-grid compact">
              <MetricCard
                label="Prefill"
                value={latency.prefill_seconds.toFixed(3) + " s"}
                note={latency.prefill_tokens_per_second.toFixed(1) + " t/s"}
              />
              <MetricCard
                label="Decode"
                value={latency.decode_seconds.toFixed(3) + " s"}
                note={latency.decode_average_tokens_per_second.toFixed(1) + " t/s"}
              />
              <MetricCard
                label="Total compute"
                value={latency.total_seconds.toFixed(3) + " s"}
              />
            </div>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Finalist validation</div>
              <h2>llama-server + SPEED-Bench</h2>
              <p className="section-copy">
                Runs the selected Candidate with its resolved placement and keeps the
                validation record append-only.
              </p>
            </div>
            <StatusBadge
              status={
                candidate.server_validation_count > 0 ? "completed" : "planned"
              }
            />
          </div>
          <div className="run-form">
            <label className="field">
              <span>llama-server</span>
              <select
                value={serverBinaryId}
                onChange={(event) => setServerBinaryId(event.target.value)}
              >
                <option value="">Choose…</option>
                {binaries
                  .filter((item) => item.kind === "llama-server")
                  .map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.path}
                    </option>
                  ))}
              </select>
            </label>
            <label className="field">
              <span>SPEED-Bench</span>
              <select
                value={speedBinaryId}
                onChange={(event) => setSpeedBinaryId(event.target.value)}
              >
                <option value="">Choose…</option>
                {binaries
                  .filter((item) => item.kind === "speed-bench")
                  .map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.path}
                    </option>
                  ))}
              </select>
            </label>
            <label className="field small-field">
              <span>Port</span>
              <input
                type="number"
                min="1"
                max="65535"
                value={validationPort}
                onChange={(event) => setValidationPort(Number(event.target.value))}
              />
            </label>
            <button
              className="button button-primary self-end"
              disabled={
                validating ||
                !profile ||
                !serverBinaryId ||
                !speedBinaryId ||
                placements.length === 0
              }
              onClick={() => void validate()}
            >
              {validating ? "Validating…" : "Validate Candidate"}
            </button>
          </div>
          {history?.benchmarks.length ? (
            <div className="candidate-table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Status</th>
                    <th>Prompt t/s</th>
                    <th>Decode t/s</th>
                    <th>Latency</th>
                    <th>Acceptance</th>
                  </tr>
                </thead>
                <tbody>
                  {history.benchmarks.map((item) => (
                    <tr key={item.id}>
                      <td>{item.category}</td>
                      <td><StatusBadge status={item.status} /></td>
                      <td>{formatMetric(item.avg_prompt_ts)}</td>
                      <td>{formatMetric(item.avg_pred_ts)}</td>
                      <td>
                        {item.avg_latency_ms === null
                          ? "—"
                          : item.avg_latency_ms.toFixed(1) + " ms"}
                      </td>
                      <td>
                        {item.accept_rate === null
                          ? "—"
                          : (item.accept_rate * 100).toFixed(1) + "%"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">No server validation records yet.</p>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <div className="eyebrow">Launcher promotion</div>
              <h2>Proposed profile patch</h2>
              <p className="section-copy">
                Generate a reproducible patch from the current launcher configuration.
                The launcher file is never changed by this action; the proposal and
                supporting validation provenance are persisted for review.
              </p>
            </div>
            <StatusBadge
              status={
                history?.evaluations.some(
                  (item) =>
                    item.stage === "server-validated" &&
                    item.decision === "completed"
                )
                  ? "server-validated"
                  : "planned"
              }
            />
          </div>
          <div className="button-row">
            <button
              className="button button-primary"
              disabled={
                promoting ||
                !profile ||
                !history?.evaluations.some(
                  (item) =>
                    item.stage === "server-validated" &&
                    item.decision === "completed"
                )
              }
              onClick={() => void promote()}
            >
              {promoting ? "Generating…" : "Generate launcher patch"}
            </button>
          </div>
          {promotion ? (
            <>
              <div className="definition-grid">
                <span>Source profile</span>
                <strong>{promotion.source_profile}</strong>
                <span>Argument changes</span>
                <strong>{promotion.changes.length}</strong>
                <span>Promotion record</span>
                <strong>{promotion.id}</strong>
              </div>
              <pre className="promotion-patch">
                {promotion.patch || "# No launcher changes are required.\n"}
              </pre>
              <JsonDetails label="Promotion provenance" value={promotion} />
            </>
          ) : (
            <p className="muted">
              A completed server validation is required before a promotion patch can
              be generated.
            </p>
          )}
        </div>
      </section>
    </main>
  );
}
