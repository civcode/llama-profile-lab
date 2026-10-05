import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { EmptyState, ErrorBanner, PageHeader } from "../components";
import type {
  BinaryRecord,
  CandidateSummary,
  Experiment,
  ModelRecord
} from "../types";

interface CandidateOption {
  experimentId: string;
  experimentName: string;
  candidate: CandidateSummary;
}

interface InstanceDraft {
  key: string;
  instanceId: string;
  role: string;
  candidateId: string;
  modelId: string;
  binaryId: string;
  serverIdentity: string;
}

const ALL_PHASES = ["dd", "pp", "pd", "dp"] as const;

function modelLabel(model: ModelRecord): string {
  const details = [model.architecture, model.quantization].filter(Boolean).join(" · ");
  const path = model.files[0]?.path ?? model.id;
  return details ? details + " · " + path : path;
}

export function NewDeploymentPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [candidateOptions, setCandidateOptions] = useState<CandidateOption[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [binaries, setBinaries] = useState<BinaryRecord[]>([]);
  const [workloadExperimentId, setWorkloadExperimentId] = useState("");
  const [instances, setInstances] = useState<InstanceDraft[]>([
    {
      key: "instance-a",
      instanceId: "model_a",
      role: "primary",
      candidateId: "",
      modelId: "",
      binaryId: "",
      serverIdentity: "model_a"
    },
    {
      key: "instance-b",
      instanceId: "model_b",
      role: "secondary",
      candidateId: "",
      modelId: "",
      binaryId: "",
      serverIdentity: "model_b"
    }
  ]);
  const [phases, setPhases] = useState<Array<(typeof ALL_PHASES)[number]>>([
    "dd",
    "pp",
    "pd",
    "dp"
  ]);
  const [hostRamMarginMiB, setHostRamMarginMiB] = useState(0);
  const [allowCpuOffload, setAllowCpuOffload] = useState(false);
  const [allowSwap, setAllowSwap] = useState(false);
  const [maximumPower, setMaximumPower] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let alive = true;
    Promise.all([api.experiments(), api.models(), api.binaries()])
      .then(async ([experimentValues, modelValues, binaryValues]) => {
        const candidates = await Promise.all(
          experimentValues.map(async (experiment) => ({
            experiment,
            candidates: await api.candidates(experiment.id)
          }))
        );
        if (!alive) return;
        const options = candidates.flatMap(({ experiment, candidates: values }) =>
          values.map((candidate) => ({
            experimentId: experiment.id,
            experimentName: experiment.name,
            candidate
          }))
        );
        const servers = binaryValues.filter((item) => item.kind === "llama-server");
        setExperiments(experimentValues);
        setCandidateOptions(options);
        setModels(modelValues);
        setBinaries(binaryValues);
        setWorkloadExperimentId(experimentValues[0]?.id ?? "");
        setInstances((current) =>
          current.map((item, index) => ({
            ...item,
            candidateId: options[index]?.candidate.id ?? options[0]?.candidate.id ?? "",
            modelId: modelValues[index]?.id ?? modelValues[0]?.id ?? "",
            binaryId: servers[index]?.id ?? servers[0]?.id ?? ""
          }))
        );
      })
      .catch((reason) => alive && setError(reason))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  const serverBinaries = useMemo(
    () => binaries.filter((item) => item.kind === "llama-server"),
    [binaries]
  );
  const workloadExperiment = experiments.find(
    (item) => item.id === workloadExperimentId
  );
  const duplicateInstanceId =
    new Set(instances.map((item) => item.instanceId.trim())).size !== instances.length;
  const incompleteInstance = instances.some(
    (item) =>
      !item.instanceId.trim() ||
      !item.role.trim() ||
      !item.candidateId ||
      !item.modelId ||
      !item.binaryId ||
      !item.serverIdentity.trim()
  );
  const validMaximumPower =
    maximumPower.trim() === "" ||
    (Number.isFinite(Number(maximumPower)) && Number(maximumPower) > 0);
  const canSave =
    instances.length >= 2 &&
    !duplicateInstanceId &&
    !incompleteInstance &&
    Boolean(workloadExperiment) &&
    phases.length > 0 &&
    validMaximumPower &&
    !saving;

  function updateInstance(index: number, patch: Partial<InstanceDraft>) {
    setInstances((current) =>
      current.map((item, itemIndex) =>
        itemIndex === index ? { ...item, ...patch } : item
      )
    );
  }

  function addInstance() {
    const ordinal = instances.length + 1;
    const server = serverBinaries[0];
    setInstances((current) => [
      ...current,
      {
        key: "instance-" + Date.now() + "-" + ordinal,
        instanceId: "model_" + ordinal,
        role: "member-" + ordinal,
        candidateId: candidateOptions[0]?.candidate.id ?? "",
        modelId: models[0]?.id ?? "",
        binaryId: server?.id ?? "",
        serverIdentity: "model_" + ordinal
      }
    ]);
  }

  function togglePhase(phase: (typeof ALL_PHASES)[number]) {
    setPhases((current) =>
      current.includes(phase)
        ? current.filter((item) => item !== phase)
        : ALL_PHASES.filter((item) => [...current, phase].includes(item))
    );
  }

  async function save() {
    if (!canSave || !workloadExperiment) return;
    try {
      setSaving(true);
      setError(null);
      const deployment = await api.createDeployment({
        deployment: {
          schema: "llama-profile-deployment-candidate",
          version: 1,
          instances: instances.map((item) => ({
            instance_id: item.instanceId.trim(),
            candidate_id: item.candidateId,
            role: item.role.trim(),
            model_artifact_id: item.modelId,
            binary_id: item.binaryId,
            requested_placement: {
              devices: null,
              n_gpu_layers: null,
              split_mode: null,
              main_gpu: null,
              tensor_split: null,
              override_tensor: []
            },
            server_identity: item.serverIdentity.trim()
          })),
          resource_policy: {
            device_memory_margin_bytes: {},
            logical_device_mappings: [],
            host_ram_margin_bytes: Math.round(hostRamMarginMiB * 1024 * 1024),
            allow_cpu_offload: allowCpuOffload,
            allow_swap: allowSwap,
            allowed_devices: [],
            allowed_backend_pairs: [],
            maximum_total_power_w:
              maximumPower.trim() === "" ? null : Number(maximumPower)
          },
          workload_mix: {
            workload_suite_id: workloadExperiment.definition.workload_suite_id,
            phases
          }
        }
      });
      window.location.hash = "#/deployments/" + deployment.id;
    } catch (reason) {
      setError(reason);
    } finally {
      setSaving(false);
    }
  }

  return (
    <main className="page narrow-page">
      <PageHeader
        eyebrow="Multi-model optimizer"
        title="New deployment"
        actions={<a className="button" href="#/deployments">Cancel</a>}
      />
      <ErrorBanner error={error} />
      {loading ? <div className="skeleton large" /> : null}
      {!loading && candidateOptions.length === 0 ? (
        <EmptyState title="No persisted Candidates are available.">
          Create and plan a V1 experiment first; its Candidates become selectable
          deployment instances without requiring manual IDs.
        </EmptyState>
      ) : null}

      {!loading && candidateOptions.length > 0 ? (
        <>
          <section className="panel">
            <div className="section-number">1</div>
            <div className="section-body">
              <div className="section-heading-row">
                <div>
                  <h2>Model instances</h2>
                  <p className="section-copy">
                    Choose persisted Candidates, registered model artifacts, and exact
                    llama-server binaries. Internal IDs remain available in advanced
                    details but are not the primary selection surface.
                  </p>
                </div>
                <button className="button" type="button" onClick={addInstance}>
                  Add instance
                </button>
              </div>
              <div className="deployment-instance-list">
                {instances.map((instance, index) => (
                  <div className="deployment-instance-card" key={instance.key}>
                    <div className="deployment-instance-heading">
                      <strong>Instance {index + 1}</strong>
                      {instances.length > 2 ? (
                        <button
                          className="button button-danger"
                          type="button"
                          onClick={() =>
                            setInstances((current) =>
                              current.filter((_, itemIndex) => itemIndex !== index)
                            )
                          }
                        >
                          Remove
                        </button>
                      ) : null}
                    </div>
                    <div className="control-grid two">
                      <label className="field">
                        <span>Instance name</span>
                        <input
                          value={instance.instanceId}
                          onChange={(event) =>
                            updateInstance(index, {
                              instanceId: event.target.value,
                              serverIdentity: event.target.value
                            })
                          }
                        />
                      </label>
                      <label className="field">
                        <span>Role</span>
                        <input
                          value={instance.role}
                          onChange={(event) =>
                            updateInstance(index, { role: event.target.value })
                          }
                        />
                      </label>
                    </div>
                    <label className="field">
                      <span>Base Candidate</span>
                      <select
                        value={instance.candidateId}
                        onChange={(event) =>
                          updateInstance(index, { candidateId: event.target.value })
                        }
                      >
                        {candidateOptions.map((option) => (
                          <option
                            key={option.experimentId + ":" + option.candidate.id}
                            value={option.candidate.id}
                          >
                            {option.experimentName} · Candidate #
                            {option.candidate.ordinal + 1} · ctx{" "}
                            {option.candidate.candidate.context.size}
                          </option>
                        ))}
                      </select>
                    </label>
                    <div className="control-grid two">
                      <label className="field">
                        <span>Model artifact</span>
                        <select
                          value={instance.modelId}
                          onChange={(event) =>
                            updateInstance(index, { modelId: event.target.value })
                          }
                        >
                          {models.map((model) => (
                            <option value={model.id} key={model.id}>
                              {modelLabel(model)}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="field">
                        <span>llama-server binary</span>
                        <select
                          value={instance.binaryId}
                          onChange={(event) =>
                            updateInstance(index, { binaryId: event.target.value })
                          }
                        >
                          {serverBinaries.map((binary) => (
                            <option value={binary.id} key={binary.id}>
                              {binary.path}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  </div>
                ))}
              </div>
              {duplicateInstanceId ? (
                <div className="banner banner-error">
                  Instance names must be unique.
                </div>
              ) : null}
            </div>
          </section>

          <section className="panel">
            <div className="section-number">2</div>
            <div className="section-body">
              <h2>Workload mix</h2>
              <p className="section-copy">
                Reuse a persisted workload suite and choose the concurrent phases to
                generate for every feasible placement.
              </p>
              <label className="field">
                <span>Workload source experiment</span>
                <select
                  value={workloadExperimentId}
                  onChange={(event) => setWorkloadExperimentId(event.target.value)}
                >
                  {experiments.map((experiment) => (
                    <option value={experiment.id} key={experiment.id}>
                      {experiment.name} · {experiment.workload_suite.cases.length} cases
                    </option>
                  ))}
                </select>
              </label>
              <div className="phase-picker">
                {ALL_PHASES.map((phase) => (
                  <label className="phase-option" key={phase}>
                    <input
                      type="checkbox"
                      checked={phases.includes(phase)}
                      onChange={() => togglePhase(phase)}
                    />
                    <span>{phase.toUpperCase()}</span>
                  </label>
                ))}
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="section-number">3</div>
            <div className="section-body">
              <h2>Host resource policy</h2>
              <p className="section-copy">
                These deployment-wide limits are applied before joint placement
                execution. Device-specific margins are configured during planning.
              </p>
              <div className="control-grid three">
                <label className="field">
                  <span>Host RAM reserve (MiB)</span>
                  <input
                    type="number"
                    min="0"
                    value={hostRamMarginMiB}
                    onChange={(event) =>
                      setHostRamMarginMiB(Number(event.target.value))
                    }
                  />
                </label>
                <label className="field">
                  <span>Maximum total power (W)</span>
                  <input
                    type="number"
                    min="1"
                    value={maximumPower}
                    placeholder="No limit"
                    onChange={(event) => setMaximumPower(event.target.value)}
                  />
                </label>
                <div className="deployment-policy-toggles">
                  <label className="phase-option">
                    <input
                      type="checkbox"
                      checked={allowCpuOffload}
                      onChange={(event) => setAllowCpuOffload(event.target.checked)}
                    />
                    <span>Allow CPU offload</span>
                  </label>
                  <label className="phase-option">
                    <input
                      type="checkbox"
                      checked={allowSwap}
                      onChange={(event) => setAllowSwap(event.target.checked)}
                    />
                    <span>Allow swap</span>
                  </label>
                </div>
              </div>
            </div>
          </section>

          <section className="save-bar">
            <div className="grow">
              <strong>
                {instances.length} instances · {phases.length} phases
              </strong>
              <div className="muted">
                The deployment is immutable after creation; search dimensions are
                configured on the deployment page before planning.
              </div>
            </div>
            <button
              className="button button-primary button-large"
              disabled={!canSave}
              onClick={() => void save()}
            >
              {saving ? "Creating…" : "Create deployment"}
            </button>
          </section>
        </>
      ) : null}
    </main>
  );
}
