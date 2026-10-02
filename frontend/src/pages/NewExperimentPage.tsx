import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { ErrorBanner, JsonDetails, PageHeader } from "../components";
import {
  candidateFromProfile,
  defaultWorkloads,
  dimensionSupported,
  parseDimensionValues,
  previewCandidateCount,
  workloadLabel
} from "../experiment";
import type {
  BinaryRecord,
  JsonScalar,
  LauncherProfile,
  ParameterDefinition,
  Placement,
  SearchDimension,
  WorkloadSuiteCase
} from "../types";

interface WorkloadDraft {
  kind:
    | "microbench-prefill"
    | "microbench-decode"
    | "microbench-combined"
    | "speed-bench";
  label: string;
  prompt: number;
  generate: number;
  depthType: "absolute" | "fraction";
  depthValue: number;
  bench: string;
  categories: string;
  outputTokens: number;
  concurrency: number;
}

function toDraft(item: WorkloadSuiteCase): WorkloadDraft {
  if (item.kind === "speed-bench") {
    return {
      kind: "speed-bench",
      label: item.label ?? "Server throughput",
      prompt: 0,
      generate: 0,
      depthType: "absolute",
      depthValue: 0,
      bench: item.speed_bench.bench,
      categories: item.speed_bench.categories.join(", "),
      outputTokens: item.speed_bench.output_tokens,
      concurrency: item.speed_bench.concurrency
    };
  }
  return {
    kind: item.kind,
    label: item.label ?? "",
    prompt:
      item.kind === "microbench-prefill" || item.kind === "microbench-combined"
        ? item.prompt_tokens
        : 0,
    generate:
      item.kind === "microbench-decode" || item.kind === "microbench-combined"
        ? item.generate_tokens
        : 0,
    depthType: item.depth.type,
    depthValue:
      item.depth.type === "absolute" ? item.depth.tokens : item.depth.value,
    bench: "throughput_1k",
    categories: "all",
    outputTokens: 256,
    concurrency: 1
  };
}

function fromDraft(item: WorkloadDraft): WorkloadSuiteCase {
  if (item.kind === "speed-bench") {
    const categories = item.categories
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
    return {
      kind: "speed-bench",
      label: item.label,
      safety_margin_tokens: 0,
      speed_bench: {
        bench: item.bench.trim() || "throughput_1k",
        categories: categories.length ? categories : ["all"],
        output_tokens: Math.max(1, Math.round(item.outputTokens)),
        concurrency: Math.max(1, Math.round(item.concurrency)),
        limit: null,
        request: {}
      }
    };
  }
  const depth =
    item.depthType === "absolute"
      ? ({ type: "absolute", tokens: Math.max(0, Math.round(item.depthValue)) } as const)
      : ({ type: "fraction", value: Math.min(1, Math.max(0, item.depthValue)) } as const);
  if (item.kind === "microbench-prefill") {
    return {
      kind: item.kind,
      label: item.label,
      safety_margin_tokens: 0,
      prompt_tokens: Math.max(1, Math.round(item.prompt)),
      depth
    };
  }
  if (item.kind === "microbench-decode") {
    return {
      kind: item.kind,
      label: item.label,
      safety_margin_tokens: 0,
      generate_tokens: Math.max(1, Math.round(item.generate)),
      depth
    };
  }
  return {
    kind: item.kind,
    label: item.label,
    safety_margin_tokens: 0,
    prompt_tokens: Math.max(1, Math.round(item.prompt)),
    generate_tokens: Math.max(1, Math.round(item.generate)),
    depth
  };
}

export function NewExperimentPage() {
  const [profiles, setProfiles] = useState<LauncherProfile[]>([]);
  const [parameters, setParameters] = useState<ParameterDefinition[]>([]);
  const [binaries, setBinaries] = useState<BinaryRecord[]>([]);
  const [placements, setPlacements] = useState<Placement[]>([]);
  const [profileId, setProfileId] = useState("");
  const [name, setName] = useState("");
  const [capabilityBinaryId, setCapabilityBinaryId] = useState("");
  const [dimensionInputs, setDimensionInputs] = useState<Record<string, string>>({
    "compute.batch_size": "2048, 4096, 8192",
    "compute.ubatch_size": "512, 1024, 2048, 4096"
  });
  const [constraint, setConstraint] = useState(
    "compute.ubatch_size <= compute.batch_size"
  );
  const [workloads, setWorkloads] = useState<WorkloadDraft[]>(
    defaultWorkloads().map(toDraft)
  );
  const [repetitions, setRepetitions] = useState(3);
  const [placementMode, setPlacementMode] = useState<"per-candidate" | "fixed">(
    "per-candidate"
  );
  const [fixedPlacementId, setFixedPlacementId] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    Promise.all([api.profiles(), api.parameters(), api.binaries(), api.placements()])
      .then(([profileResponse, parameterItems, binaryItems, placementItems]) => {
        setProfiles(profileResponse.items);
        setParameters(parameterItems);
        setBinaries(binaryItems);
        setPlacements(placementItems);
        const first = profileResponse.items[0];
        if (first) {
          setProfileId(first.id);
          setName(first.id + " tuning");
        }
        const bench = binaryItems.find((item) => item.kind === "llama-bench");
        if (bench) setCapabilityBinaryId(bench.id);
      })
      .catch(setError);
  }, []);

  const profile = profiles.find((item) => item.id === profileId) ?? null;
  const baseCandidate = profile ? candidateFromProfile(profile) : null;
  const capabilityBinary =
    binaries.find((item) => item.id === capabilityBinaryId) ?? null;

  const dimensions = useMemo(() => {
    const values: SearchDimension[] = [];
    for (const definition of parameters) {
      const raw = dimensionInputs[definition.path];
      if (!raw) continue;
      try {
        values.push({
          path: definition.path,
          values: parseDimensionValues(raw, definition),
          condition: null
        });
      } catch {
        // Validation detail is presented when saving.
      }
    }
    return values;
  }, [dimensionInputs, parameters]);

  const preview = useMemo(() => {
    if (!baseCandidate || dimensions.length === 0) {
      return { raw: 0, valid: 0, rejected: 0 };
    }
    return previewCandidateCount(
      baseCandidate,
      dimensions,
      constraint.trim() ? [constraint.trim()] : []
    );
  }, [baseCandidate, dimensions, constraint]);

  function toggleDimension(definition: ParameterDefinition) {
    setDimensionInputs((current) => {
      if (current[definition.path] !== undefined) {
        const next = { ...current };
        delete next[definition.path];
        return next;
      }
      const currentValue = baseCandidate
        ? definition.path.split(".").reduce<unknown>(
            (value, segment) =>
              value && typeof value === "object"
                ? (value as Record<string, unknown>)[segment]
                : undefined,
            baseCandidate
          )
        : undefined;
      return {
        ...current,
        [definition.path]:
          currentValue === undefined || currentValue === null
            ? ""
            : String(currentValue)
      };
    });
  }

  function updateWorkload(index: number, patch: Partial<WorkloadDraft>) {
    setWorkloads((current) =>
      current.map((item, itemIndex) =>
        itemIndex === index ? { ...item, ...patch } : item
      )
    );
  }

  async function saveAndPlan() {
    if (!profile || !baseCandidate) {
      setError(new Error("Choose a launcher profile first."));
      return;
    }
    if (!name.trim()) {
      setError(new Error("Experiment name is required."));
      return;
    }
    if (dimensions.length === 0) {
      setError(new Error("Select at least one parameter dimension."));
      return;
    }
    if (workloads.length === 0) {
      setError(new Error("Select at least one workload."));
      return;
    }

    try {
      setSaving(true);
      setError(null);
      for (const definition of parameters) {
        const raw = dimensionInputs[definition.path];
        if (raw !== undefined) parseDimensionValues(raw, definition);
      }
      const created = await api.createExperiment({
        name: name.trim(),
        base_candidate: baseCandidate,
        search_space: {
          schema: "llama-search-space",
          version: 1,
          dimensions,
          constraints: constraint.trim() ? [constraint.trim()] : [],
          strategy: { type: "grid" }
        },
        workload_suite: {
          schema: "llama-workload-suite",
          version: 1,
          id:
            profile.id.replace(/[^A-Za-z0-9_.-]+/g, "-") +
            "-ui-" +
            Date.now().toString(36),
          description: "Created from the llama-profile-lab browser UI",
          cases: workloads.map(fromDraft)
        },
        measurement_policy: {
          schema: "llama-measurement-policy",
          version: 1,
          warmup: true,
          repetitions: Math.max(1, Math.round(repetitions)),
          delay_seconds: 0,
          adaptive: null
        },
        placement_policy:
          placementMode === "fixed"
            ? { type: "fixed", placement_id: fixedPlacementId }
            : { type: "per-candidate" },
        baseline: { type: "base-candidate" }
      });
      await api.planExperiment(created.id);
      window.location.hash = "#/experiments/" + created.id;
    } catch (reason) {
      setError(reason);
    } finally {
      setSaving(false);
    }
  }

  const microbenchCount = workloads.filter(
    (item) => item.kind !== "speed-bench"
  ).length;
  const serverWorkloadCount = workloads.length - microbenchCount;
  const sampleCount = preview.valid * microbenchCount * repetitions;

  return (
    <main className="page narrow-page">
      <PageHeader
        eyebrow="New experiment"
        title="Tune a production profile"
        actions={
          <a className="button" href="#/">
            Cancel
          </a>
        }
      />
      <ErrorBanner error={error} />

      <section className="panel">
        <div className="section-number">1</div>
        <div className="section-body">
          <h2>Source profile</h2>
          <p className="section-copy">
            Start from the effective production settings. The experiment will freeze its
            own immutable Candidate.
          </p>
          <label className="field">
            <span>Launcher profile</span>
            <select
              value={profileId}
              onChange={(event) => {
                setProfileId(event.target.value);
                setName(event.target.value + " tuning");
              }}
            >
              <option value="">Choose a profile…</option>
              {profiles.map((item) => (
                <option value={item.id} key={item.id}>
                  {item.id}
                </option>
              ))}
            </select>
          </label>
          {profile ? (
            <div className="profile-summary">
              <div>
                <span className="field-label">Model</span>
                <code>{profile.model_path}</code>
              </div>
              <div>
                <span className="field-label">Profile chain</span>
                <span>{profile.profiles.join(" → ") || "model settings only"}</span>
              </div>
              <JsonDetails label="Resolved launcher arguments" value={profile.args} />
            </div>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <div className="section-number">2</div>
        <div className="section-body">
          <h2>Vary parameters</h2>
          <p className="section-copy">
            Check dimensions to sweep. Unsupported options are visibly disabled when a
            registered llama-bench binary is selected.
          </p>
          <label className="field">
            <span>Capability target</span>
            <select
              value={capabilityBinaryId}
              onChange={(event) => setCapabilityBinaryId(event.target.value)}
            >
              <option value="">No binary selected</option>
              {binaries
                .filter((item) => item.kind === "llama-bench")
                .map((item) => (
                  <option value={item.id} key={item.id}>
                    {item.path}
                  </option>
                ))}
            </select>
          </label>
          <div className="dimension-list">
            {parameters.map((definition) => {
              const enabled = dimensionInputs[definition.path] !== undefined;
              const supported = dimensionSupported(definition, capabilityBinary);
              return (
                <div
                  className={"dimension-row " + (!supported ? "is-disabled" : "")}
                  key={definition.path}
                >
                  <label className="dimension-check">
                    <input
                      type="checkbox"
                      checked={enabled}
                      disabled={!supported}
                      onChange={() => toggleDimension(definition)}
                    />
                    <span>
                      <strong>{definition.label}</strong>
                      <small>{definition.path}</small>
                    </span>
                  </label>
                  <input
                    aria-label={definition.label + " values"}
                    value={dimensionInputs[definition.path] ?? ""}
                    disabled={!enabled || !supported}
                    placeholder={
                      definition.string_choices?.join(", ") ?? "comma-separated values"
                    }
                    onChange={(event) =>
                      setDimensionInputs((current) => ({
                        ...current,
                        [definition.path]: event.target.value
                      }))
                    }
                  />
                  {!supported ? (
                    <span className="support-note">
                      {definition.cli_argument} unavailable
                    </span>
                  ) : (
                    <span className="support-note">
                      {definition.affects_placement ? "affects placement" : definition.category}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          <label className="field">
            <span>Constraint</span>
            <input
              value={constraint}
              onChange={(event) => setConstraint(event.target.value)}
              placeholder="compute.ubatch_size <= compute.batch_size"
            />
          </label>
        </div>
      </section>

      <section className="panel">
        <div className="section-number">3</div>
        <div className="section-body">
          <div className="section-heading-row">
            <div>
              <h2>Workloads</h2>
              <p className="section-copy">
                The reference preset keeps prefill and decode separate and tests decode
                at both shallow and realistic active context.
              </p>
            </div>
            <div className="button-row">
              <button
                className="button"
                type="button"
                onClick={() =>
                  setWorkloads((current) => [
                    ...current,
                    {
                      kind: "microbench-decode",
                      label: "New decode workload",
                      prompt: 0,
                      generate: 256,
                      depthType: "absolute",
                      depthValue: 4096,
                      bench: "throughput_1k",
                      categories: "all",
                      outputTokens: 256,
                      concurrency: 1
                    }
                  ])
                }
              >
                Add microbenchmark
              </button>
              <button
                className="button"
                type="button"
                onClick={() =>
                  setWorkloads((current) => [
                    ...current,
                    {
                      kind: "speed-bench",
                      label: "Server throughput",
                      prompt: 0,
                      generate: 0,
                      depthType: "absolute",
                      depthValue: 0,
                      bench: "throughput_1k",
                      categories: "all",
                      outputTokens: 256,
                      concurrency: 1
                    }
                  ])
                }
              >
                Add server validation
              </button>
            </div>
          </div>
          <div className="workload-list">
            {workloads.map((item, index) => (
              <div className="workload-row" key={index}>
                <input
                  aria-label={"Workload " + (index + 1) + " label"}
                  value={item.label}
                  onChange={(event) =>
                    updateWorkload(index, { label: event.target.value })
                  }
                />
                <select
                  value={item.kind}
                  onChange={(event) =>
                    updateWorkload(index, {
                      kind: event.target.value as WorkloadDraft["kind"]
                    })
                  }
                >
                  <option value="microbench-prefill">Prefill</option>
                  <option value="microbench-decode">Decode</option>
                  <option value="microbench-combined">Combined</option>
                  <option value="speed-bench">SPEED-Bench</option>
                </select>
                {item.kind === "speed-bench" ? (
                  <>
                    <label className="compact-field">
                      <span>Bench</span>
                      <input
                        value={item.bench}
                        onChange={(event) =>
                          updateWorkload(index, { bench: event.target.value })
                        }
                      />
                    </label>
                    <label className="compact-field">
                      <span>Output tokens</span>
                      <input
                        type="number"
                        min="1"
                        value={item.outputTokens}
                        onChange={(event) =>
                          updateWorkload(index, {
                            outputTokens: Number(event.target.value)
                          })
                        }
                      />
                    </label>
                    <label className="compact-field">
                      <span>Concurrency</span>
                      <input
                        type="number"
                        min="1"
                        value={item.concurrency}
                        onChange={(event) =>
                          updateWorkload(index, {
                            concurrency: Number(event.target.value)
                          })
                        }
                      />
                    </label>
                    <label className="compact-field">
                      <span>Categories</span>
                      <input
                        value={item.categories}
                        onChange={(event) =>
                          updateWorkload(index, { categories: event.target.value })
                        }
                        placeholder="all"
                      />
                    </label>
                  </>
                ) : null}
                {item.kind !== "speed-bench" && item.kind !== "microbench-decode" ? (
                  <label className="compact-field">
                    <span>Prompt</span>
                    <input
                      type="number"
                      min="1"
                      value={item.prompt}
                      onChange={(event) =>
                        updateWorkload(index, { prompt: Number(event.target.value) })
                      }
                    />
                  </label>
                ) : null}
                {item.kind !== "speed-bench" && item.kind !== "microbench-prefill" ? (
                  <label className="compact-field">
                    <span>Generate</span>
                    <input
                      type="number"
                      min="1"
                      value={item.generate}
                      onChange={(event) =>
                        updateWorkload(index, {
                          generate: Number(event.target.value)
                        })
                      }
                    />
                  </label>
                ) : null}
                {item.kind !== "speed-bench" ? (
                  <label className="compact-field">
                  <span>Depth</span>
                  <div className="inline-inputs">
                    <select
                      value={item.depthType}
                      onChange={(event) =>
                        updateWorkload(index, {
                          depthType: event.target.value as "absolute" | "fraction"
                        })
                      }
                    >
                      <option value="absolute">tokens</option>
                      <option value="fraction">fraction</option>
                    </select>
                    <input
                      type="number"
                      min="0"
                      step={item.depthType === "fraction" ? "0.05" : "1"}
                      max={item.depthType === "fraction" ? "1" : undefined}
                      value={item.depthValue}
                      onChange={(event) =>
                        updateWorkload(index, {
                          depthValue: Number(event.target.value)
                        })
                      }
                    />
                  </div>
                </label>
                ) : null}
                <button
                  className="icon-button"
                  aria-label={"Remove " + workloadLabel(fromDraft(item), index)}
                  type="button"
                  onClick={() =>
                    setWorkloads((current) =>
                      current.filter((_, itemIndex) => itemIndex !== index)
                    )
                  }
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="section-number">4</div>
        <div className="section-body split-fields">
          <div>
            <h2>Measurement</h2>
            <label className="field">
              <span>Timed repetitions</span>
              <input
                type="number"
                min="1"
                value={repetitions}
                onChange={(event) => setRepetitions(Number(event.target.value))}
              />
            </label>
          </div>
          <div>
            <h2>Placement</h2>
            <label className="field">
              <span>Strategy</span>
              <select
                value={placementMode}
                onChange={(event) =>
                  setPlacementMode(
                    event.target.value as "per-candidate" | "fixed"
                  )
                }
              >
                <option value="per-candidate">Re-fit each candidate</option>
                <option value="fixed">Reuse fixed placement</option>
              </select>
            </label>
            {placementMode === "fixed" ? (
              <label className="field">
                <span>Resolved placement</span>
                <select
                  value={fixedPlacementId}
                  onChange={(event) => setFixedPlacementId(event.target.value)}
                >
                  <option value="">Choose a placement…</option>
                  {placements.map((item) => (
                    <option value={item.id} key={item.id}>
                      {item.id} · ctx {item.production_context_size} · ngl{" "}
                      {item.n_gpu_layers}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
          </div>
        </div>
      </section>

      <section className="plan-preview">
        <div>
          <div className="eyebrow">Plan preview</div>
          <h2>{preview.valid} valid candidates</h2>
          <p>
            {preview.raw} raw combinations · {preview.rejected} rejected by the
            current constraint
          </p>
        </div>
        <div className="plan-numbers">
          <div>
            <strong>{preview.valid * microbenchCount}</strong>
            <span>benchmark cases</span>
          </div>
          <div>
            <strong>{sampleCount}</strong>
            <span>timed samples</span>
          </div>
          <div>
            <strong>{serverWorkloadCount}</strong>
            <span>server workload{serverWorkloadCount === 1 ? "" : "s"}</span>
          </div>
        </div>
      </section>

      <section className="save-bar">
        <label className="field grow">
          <span>Experiment name</span>
          <input value={name} onChange={(event) => setName(event.target.value)} />
        </label>
        <button
          className="button button-primary button-large"
          disabled={
            saving ||
            preview.valid === 0 ||
            (placementMode === "fixed" && !fixedPlacementId)
          }
          onClick={saveAndPlan}
        >
          {saving ? "Saving…" : "Save + plan experiment"}
        </button>
      </section>
    </main>
  );
}
