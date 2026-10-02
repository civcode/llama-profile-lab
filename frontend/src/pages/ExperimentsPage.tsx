import { useEffect, useState } from "react";
import { api } from "../api";
import { ErrorBanner, PageHeader, StatusBadge } from "../components";
import type { Experiment, LauncherProfile } from "../types";

function profileName(experiment: Experiment, profiles: LauncherProfile[]): string {
  const target = experiment.base_candidate.model.target_model_id;
  const prefix = "launcher-profile:";
  if (target.startsWith(prefix)) {
    const id = target.slice(prefix.length);
    return profiles.find((item) => item.id === id)?.id ?? id;
  }
  return target;
}

export function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [profiles, setProfiles] = useState<LauncherProfile[]>([]);
  const [validated, setValidated] = useState<Record<string, number>>({});
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    Promise.all([api.experiments(), api.profiles()])
      .then(async ([items, profileResponse]) => {
        if (!alive) return;
        setExperiments(items);
        setProfiles(profileResponse.items);
        const pairs = await Promise.all(
          items.map(async (experiment) => {
            if (experiment.candidate_count === 0) return [experiment.id, 0] as const;
            const candidates = await api.candidates(experiment.id);
            return [
              experiment.id,
              candidates.filter((candidate) => candidate.server_validation_count > 0)
                .length
            ] as const;
          })
        );
        if (alive) setValidated(Object.fromEntries(pairs));
      })
      .catch((reason) => alive && setError(reason))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <main className="page">
      <PageHeader
        eyebrow="Workspace"
        title="Experiments"
        actions={
          <a className="button button-primary" href="#/new">
            New experiment
          </a>
        }
      />
      <ErrorBanner error={error} />
      {loading ? <div className="skeleton large" /> : null}
      {!loading && experiments.length === 0 ? (
        <div className="empty-state large">
          <strong>No experiments yet.</strong>
          <div>Start from a launcher profile and build a reproducible search plan.</div>
          <a className="button button-primary" href="#/new">
            Create the first experiment
          </a>
        </div>
      ) : null}
      <div className="experiment-grid">
        {experiments.map((experiment) => {
          const completed =
            experiment.benchmark_case_count - experiment.incomplete_case_count;
          const validationCount = validated[experiment.id] ?? 0;
          return (
            <a
              href={"#/experiments/" + experiment.id}
              className="experiment-card"
              key={experiment.id}
            >
              <div className="card-topline">
                <StatusBadge status={experiment.status} />
                <span className="muted">
                  {new Date(experiment.created_at).toLocaleString()}
                </span>
              </div>
              <h2>{experiment.name}</h2>
              <div className="profile-line">{profileName(experiment, profiles)}</div>
              <div className="progress-line">
                <div className="progress-track" aria-label="Experiment progress">
                  <span
                    style={{
                      width:
                        experiment.benchmark_case_count > 0
                          ? Math.round(
                              (completed / experiment.benchmark_case_count) * 100
                            ) + "%"
                          : "0%"
                    }}
                  />
                </div>
                <strong>
                  {completed}/{experiment.benchmark_case_count}
                </strong>
              </div>
              <div className="card-meta">
                <span>{experiment.candidate_count} candidates</span>
                <span>
                  Baseline:{" "}
                  {experiment.definition.baseline.type === "base-candidate"
                    ? "base profile"
                    : "selected candidate"}
                </span>
                <span>
                  {validationCount > 0
                    ? validationCount + " server validated"
                    : "Not server validated"}
                </span>
              </div>
            </a>
          );
        })}
      </div>
    </main>
  );
}
