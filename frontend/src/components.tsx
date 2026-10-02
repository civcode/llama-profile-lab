import type { ReactNode } from "react";
import type { JsonScalar, MatrixProjection, ParetoResult } from "./types";

export function StatusBadge({ status }: { status: string }) {
  const normalized = status.replaceAll("_", " ");
  return <span className={"status status-" + status}>{normalized}</span>;
}

export function PageHeader({
  eyebrow,
  title,
  actions
}: {
  eyebrow?: string;
  title: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
        <h1>{title}</h1>
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  );
}

export function MetricCard({
  label,
  value,
  note
}: {
  label: string;
  value: ReactNode;
  note?: string;
}) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {note ? <div className="metric-note">{note}</div> : null}
    </div>
  );
}

export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <div className="banner banner-error" role="alert">
      {error instanceof Error ? error.message : String(error)}
    </div>
  );
}

export function EmptyState({
  title,
  children
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      {children ? <div>{children}</div> : null}
    </div>
  );
}

function scalarKey(value: JsonScalar): string {
  return value === null ? "null" : String(value);
}

export function HeatmapTable({ matrix }: { matrix: MatrixProjection }) {
  const values = matrix.facets.flatMap((facet) => facet.cells.map((cell) => cell.value));
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 0;
  const intensity = (value: number) =>
    max === min ? 0.45 : 0.14 + ((value - min) / (max - min)) * 0.72;

  return (
    <div className="heatmap-stack">
      {matrix.facets.map((facet, facetIndex) => {
        const cells = new Map(
          facet.cells.map((cell) => [
            scalarKey(cell.x) + "::" + scalarKey(cell.y),
            cell
          ])
        );
        return (
          <div className="heatmap-wrap" key={facetIndex}>
            {matrix.facet_path ? (
              <div className="facet-title">
                {matrix.facet_path} = {scalarKey(facet.value)}
              </div>
            ) : null}
            <table className="heatmap">
              <thead>
                <tr>
                  <th>{matrix.y_path} ↓ / {matrix.x_path} →</th>
                  {matrix.x_values.map((value) => (
                    <th key={scalarKey(value)}>{scalarKey(value)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.y_values.map((y) => (
                  <tr key={scalarKey(y)}>
                    <th>{scalarKey(y)}</th>
                    {matrix.x_values.map((x) => {
                      const cell = cells.get(scalarKey(x) + "::" + scalarKey(y));
                      return (
                        <td
                          key={scalarKey(x)}
                          style={
                            cell
                              ? {
                                  background: "rgba(37, 99, 235, " +
                                    intensity(cell.value).toFixed(3) +
                                    ")"
                                }
                              : undefined
                          }
                          title={
                            cell
                              ? cell.value.toFixed(4) +
                                " · " +
                                cell.sample_count +
                                " samples"
                              : "No observation"
                          }
                        >
                          {cell ? cell.value.toFixed(2) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
    </div>
  );
}

export function ParetoPlot({ result }: { result: ParetoResult }) {
  if (result.objectives.length < 2 || result.frontier.length === 0) {
    return <EmptyState title="Pareto plot needs two populated objectives." />;
  }
  const [xObjective, yObjective] = result.objectives;
  const xs = result.frontier.map((item) => item.values[xObjective.key]);
  const ys = result.frontier.map((item) => item.values[yObjective.key]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scale = (value: number, min: number, max: number, start: number, end: number) =>
    max === min ? (start + end) / 2 : start + ((value - min) / (max - min)) * (end - start);

  return (
    <div className="pareto-plot">
      <svg viewBox="0 0 520 300" role="img" aria-label="Pareto frontier scatter plot">
        <line x1="52" y1="252" x2="492" y2="252" className="axis" />
        <line x1="52" y1="24" x2="52" y2="252" className="axis" />
        <text x="272" y="288" textAnchor="middle" className="axis-label">
          {xObjective.key}
        </text>
        <text
          x="18"
          y="140"
          textAnchor="middle"
          className="axis-label"
          transform="rotate(-90 18 140)"
        >
          {yObjective.key}
        </text>
        {result.frontier.map((item) => {
          const x = scale(item.values[xObjective.key], minX, maxX, 70, 475);
          const y = scale(item.values[yObjective.key], minY, maxY, 235, 40);
          return (
            <g key={item.candidate_id}>
              <circle cx={x} cy={y} r="7" className="pareto-point" />
              <text x={x + 10} y={y - 8} className="point-label">
                #{item.candidate_ordinal + 1}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="plot-caption">
        Non-dominated configurations only. Axes use the objectives exactly as selected.
      </div>
    </div>
  );
}

export function JsonDetails({
  label = "Advanced details",
  value
}: {
  label?: string;
  value: unknown;
}) {
  return (
    <details className="json-details">
      <summary>{label}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}
