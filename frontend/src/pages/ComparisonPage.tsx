import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { EmptyState, ErrorBanner, PageHeader } from "../components";
import { workloadLabel } from "../experiment";
import type {
  CandidateComparison,
  CandidateSummary,
  Experiment,
  MetricDefinition
} from "../types";

function candidateCoordinate(
  candidate: CandidateSummary,
  path: string
): string {
  let current: unknown = candidate.candidate;
  for (const segment of path.split(".")) {
    current =
      current && typeof current === "object"
        ? (current as Record<string, unknown>)[segment]
        : undefined;
  }
  return String(current ?? "—");
}

function percent(value: number | null): string {
  if (value === null) return "—";
  return (value >= 0 ? "+" : "") + value.toFixed(2) + "%";
}

export function ComparisonPage({ experimentId }: { experimentId: string }) {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [candidates, setCandidates] = useState<CandidateSummary[]>([]);
  const [metrics, setMetrics] = useState<MetricDefinition[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [metric, setMetric] = useState("throughput.median");
  const [comparisons, setComparisons] = useState<CandidateComparison[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    Promise.all([
      api.experiment(experimentId),
      api.candidates(experimentId),
      api.metrics()
    ])
      .then(([experimentValue, candidateValues, metricValues]) => {
        setExperiment(experimentValue);
        setCandidates(candidateValues);
        setMetrics(metricValues);
        const complete = candidateValues.filter(
          (item) => item.completed_case_count > 0
        );
        setSelected(
          (complete.length >= 2 ? complete : candidateValues)
            .slice(0, Math.min(3, candidateValues.length))
            .map((item) => item.id)
        );
      })
      .catch(setError);
  }, [experimentId]);

  function toggle(candidateId: string) {
    setSelected((current) => {
      if (current.includes(candidateId)) {
        return current.filter((item) => item !== candidateId);
      }
      if (current.length >= 5) return current;
      return [...current, candidateId];
    });
  }

  async function compare() {
    if (selected.length < 2) {
      setError(new Error("Select at least two Candidates to compare."));
      return;
    }
    try {
      setLoading(true);
      setError(null);
      setComparisons(
        await Promise.all(
          selected.map((candidateId) =>
            api.compare(experimentId, candidateId, [metric])
          )
        )
      );
    } catch (reason) {
      setError(reason);
    } finally {
      setLoading(false);
    }
  }

  const workloadIndexes = useMemo(() => {
    const values = new Set<number>();
    for (const comparison of comparisons) {
      for (const delta of comparison.deltas) {
        values.add(delta.suite_case_index);
      }
    }
    return [...values].sort((a, b) => a - b);
  }, [comparisons]);

  if (!experiment) {
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
        eyebrow="Candidate comparison"
        title={experiment.name}
        actions={
          <a className="button" href={"#/experiments/" + experiment.id}>
            Back to experiment
          </a>
        }
      />
      <ErrorBanner error={error} />

      <section className="panel">
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <h2>Select 2–5 Candidates</h2>
              <p className="section-copy">
                Values are compared against the experiment baseline. The browser does
                not invent an overall winner or collapse workloads into one score.
              </p>
            </div>
            <button
              className="button button-primary"
              disabled={loading || selected.length < 2}
              onClick={() => void compare()}
            >
              {loading ? "Comparing…" : "Compare selected"}
            </button>
          </div>
          <label className="field comparison-metric">
            <span>Metric</span>
            <select value={metric} onChange={(event) => setMetric(event.target.value)}>
              {metrics.map((item) => (
                <option value={item.name} key={item.name}>
                  {item.label} {item.unit ? "(" + item.unit + ")" : ""}
                </option>
              ))}
            </select>
          </label>
          <div className="comparison-candidates">
            {candidates.map((item) => {
              const checked = selected.includes(item.id);
              const disabled = !checked && selected.length >= 5;
              return (
                <label
                  className={
                    "comparison-candidate " +
                    (checked ? "is-selected" : "") +
                    (disabled ? " is-disabled" : "")
                  }
                  key={item.id}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={disabled}
                    onChange={() => toggle(item.id)}
                  />
                  <span>
                    <strong>Candidate #{item.ordinal + 1}</strong>
                    <small>
                      {experiment.search_space.dimensions
                        .map(
                          (dimension) =>
                            dimension.path.split(".").at(-1) +
                            "=" +
                            candidateCoordinate(item, dimension.path)
                        )
                        .join(" · ")}
                    </small>
                    <small>
                      {item.completed_case_count}/{item.benchmark_case_count} completed
                    </small>
                  </span>
                </label>
              );
            })}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="section-body">
          <div className="eyebrow">Baseline-relative results</div>
          <h2>{metrics.find((item) => item.name === metric)?.label ?? metric}</h2>
          {comparisons.length ? (
            <div className="candidate-table-wrap">
              <table className="data-table comparison-table">
                <thead>
                  <tr>
                    <th>Candidate</th>
                    {workloadIndexes.map((index) => (
                      <th key={index}>
                        {workloadLabel(experiment.workload_suite.cases[index], index)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {comparisons.map((comparison) => {
                    const candidate = candidates.find(
                      (item) => item.id === comparison.candidate_id
                    );
                    return (
                      <tr key={comparison.candidate_id}>
                        <th>#{(candidate?.ordinal ?? 0) + 1}</th>
                        {workloadIndexes.map((index) => {
                          const delta = comparison.deltas.find(
                            (item) =>
                              item.suite_case_index === index &&
                              item.metric === metric
                          );
                          return (
                            <td key={index}>
                              <strong>
                                {delta?.candidate_value === null ||
                                delta?.candidate_value === undefined
                                  ? "—"
                                  : delta.candidate_value.toFixed(2)}
                              </strong>
                              <small className="comparison-delta">
                                {percent(delta?.percent_delta ?? null)}
                              </small>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState title="Choose Candidates and run the comparison.">
              Signed deltas remain workload-specific and preserve the configured
              experiment baseline.
            </EmptyState>
          )}
        </div>
      </section>
    </main>
  );
}
