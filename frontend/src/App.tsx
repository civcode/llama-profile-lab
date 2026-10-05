import { useEffect, useState } from "react";
import { CandidatePage } from "./pages/CandidatePage";
import { ComparisonPage } from "./pages/ComparisonPage";
import { DeploymentCandidatePage } from "./pages/DeploymentCandidatePage";
import { DeploymentPage } from "./pages/DeploymentPage";
import { DeploymentsPage } from "./pages/DeploymentsPage";
import { ExperimentPage } from "./pages/ExperimentPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { NewDeploymentPage } from "./pages/NewDeploymentPage";
import { NewExperimentPage } from "./pages/NewExperimentPage";

type Route =
  | { kind: "experiments" }
  | { kind: "new" }
  | { kind: "deployments" }
  | { kind: "newDeployment" }
  | { kind: "deployment"; deploymentId: string }
  | { kind: "deploymentCandidate"; deploymentId: string; candidateId: string }
  | { kind: "experiment"; experimentId: string }
  | { kind: "comparison"; experimentId: string }
  | { kind: "candidate"; experimentId: string; candidateId: string };

function parseRoute(hash: string): Route {
  const value = hash.replace(/^#\/?/, "");
  if (!value) return { kind: "experiments" };
  if (value === "new") return { kind: "new" };
  if (value === "deployments") return { kind: "deployments" };
  if (value === "deployments/new") return { kind: "newDeployment" };
  const parts = value.split("/").filter(Boolean);
  if (parts[0] === "deployments" && parts[1]) {
    if (parts[2] === "candidates" && parts[3]) {
      return {
        kind: "deploymentCandidate",
        deploymentId: parts[1],
        candidateId: parts[3]
      };
    }
    return { kind: "deployment", deploymentId: parts[1] };
  }
  if (parts[0] === "experiments" && parts[1]) {
    if (parts[2] === "compare") {
      return { kind: "comparison", experimentId: parts[1] };
    }
    if (parts[2] === "candidates" && parts[3]) {
      return {
        kind: "candidate",
        experimentId: parts[1],
        candidateId: parts[3]
      };
    }
    return { kind: "experiment", experimentId: parts[1] };
  }
  return { kind: "experiments" };
}

export function App() {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));

  useEffect(() => {
    const listener = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", listener);
    return () => window.removeEventListener("hashchange", listener);
  }, []);

  return (
    <>
      <nav className="app-nav">
        <a className="brand" href="#/">
          <span className="brand-mark">λ</span>
          <span>
            llama-profile-lab
            <small>benchmark control room</small>
          </span>
        </a>
        <div className="nav-links">
          <a href="#/">Experiments</a>
          <a href="#/new">New experiment</a>
          <a href="#/deployments">Deployments</a>
          <a href="/api/docs" target="_blank" rel="noreferrer">
            API
          </a>
        </div>
      </nav>
      {route.kind === "experiments" ? <ExperimentsPage /> : null}
      {route.kind === "new" ? <NewExperimentPage /> : null}
      {route.kind === "deployments" ? <DeploymentsPage /> : null}
      {route.kind === "newDeployment" ? <NewDeploymentPage /> : null}
      {route.kind === "deployment" ? (
        <DeploymentPage deploymentId={route.deploymentId} />
      ) : null}
      {route.kind === "deploymentCandidate" ? (
        <DeploymentCandidatePage
          deploymentId={route.deploymentId}
          candidateId={route.candidateId}
        />
      ) : null}
      {route.kind === "experiment" ? (
        <ExperimentPage experimentId={route.experimentId} />
      ) : null}
      {route.kind === "comparison" ? (
        <ComparisonPage experimentId={route.experimentId} />
      ) : null}
      {route.kind === "candidate" ? (
        <CandidatePage
          experimentId={route.experimentId}
          candidateId={route.candidateId}
        />
      ) : null}
      <footer className="app-footer">
        Local-first · SQLite is authoritative · failures remain visible
      </footer>
    </>
  );
}

export { parseRoute };
