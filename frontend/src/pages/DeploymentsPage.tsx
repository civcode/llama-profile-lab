import { useEffect, useState } from "react";
import { api } from "../api";
import { EmptyState, ErrorBanner, PageHeader, StatusBadge } from "../components";
import type { Deployment } from "../types";

export function DeploymentsPage() {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    api.deployments()
      .then((items) => alive && setDeployments(items))
      .catch((reason) => alive && setError(reason))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <main className="page">
      <PageHeader
        eyebrow="Multi-model optimizer"
        title="Deployments"
        actions={
          <a className="button button-primary" href="#/deployments/new">
            New deployment
          </a>
        }
      />
      <ErrorBanner error={error} />
      {loading ? <div className="skeleton large" /> : null}
      {!loading && deployments.length === 0 ? (
        <EmptyState title="No deployments yet.">
          Combine two or more persisted Candidates into a joint GPU placement plan.
        </EmptyState>
      ) : null}
      <div className="experiment-grid">
        {deployments.map((deployment) => (
          <a
            href={"#/deployments/" + deployment.id}
            className="experiment-card"
            key={deployment.id}
          >
            <div className="card-topline">
              <StatusBadge status={deployment.status} />
              <span className="muted">
                {new Date(deployment.created_at).toLocaleString()}
              </span>
            </div>
            <h2>
              {deployment.definition.instances
                .map((item) => item.role || item.instance_id)
                .join(" + ")}
            </h2>
            <div className="profile-line">
              {deployment.definition.instances.length} model instances ·{" "}
              {deployment.definition.workload_mix.phases
                .map((phase) => phase.toUpperCase())
                .join("/")}
            </div>
            <div className="card-meta">
              <span>{deployment.plan_count} planning pass{deployment.plan_count === 1 ? "" : "es"}</span>
              <span>{deployment.placement_count} feasible placements</span>
              <span>{deployment.run_count} deployment runs</span>
            </div>
          </a>
        ))}
      </div>
    </main>
  );
}
