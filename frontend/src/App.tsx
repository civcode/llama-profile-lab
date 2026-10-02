import { useEffect, useState } from "react";
import { CandidatePage } from "./pages/CandidatePage";
import { ExperimentPage } from "./pages/ExperimentPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { NewExperimentPage } from "./pages/NewExperimentPage";

type Route =
  | { kind: "experiments" }
  | { kind: "new" }
  | { kind: "experiment"; experimentId: string }
  | { kind: "candidate"; experimentId: string; candidateId: string };

function parseRoute(hash: string): Route {
  const value = hash.replace(/^#\/?/, "");
  if (!value) return { kind: "experiments" };
  if (value === "new") return { kind: "new" };
  const parts = value.split("/").filter(Boolean);
  if (parts[0] === "experiments" && parts[1]) {
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
          <a href="/api/docs" target="_blank" rel="noreferrer">
            API
          </a>
        </div>
      </nav>
      {route.kind === "experiments" ? <ExperimentsPage /> : null}
      {route.kind === "new" ? <NewExperimentPage /> : null}
      {route.kind === "experiment" ? (
        <ExperimentPage experimentId={route.experimentId} />
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
