import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { EmptyState, ErrorBanner, PageHeader } from "../components";
import type {
  BinaryRecord,
  CandidateSummary,
  DeviceInventoryResponse,
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
  const [deviceInventories, setDeviceInventories] = useState<Record<string, DeviceInventoryResponse>>({});
  const [selectedDevices, setSelectedDevices] = useState<string[]>([]);
  const [deviceMarginsMiB, setDeviceMarginsMiB] = useState<Record<string, string>>({});
  const [discoveringDevices, setDiscoveringDevices] = useState(false);
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
  const discoveredDevices = useMemo(() => {
    const grouped = new Map<
      string,
      {
        id: string;
        label: string;
        totalMemoryBytes: number | null;
        freeMemoryBytes: number | null;
        mappings: {
          binary_id: string;
          logical_device_name: string;
          device_id: string;
        }[];
      }
    >();
    for (const inventory of Object.values(deviceInventories)) {
      for (const device of inventory.items) {
        const stableId = device.physical_device_key ?? device.logical_device_name;
        const current = grouped.get(stableId) ?? {
          id: stableId,
          label:
            [device.vendor, device.product_name].filter(Boolean).join(" ") ||
            device.logical_device_name,
          totalMemoryBytes: device.total_memory_bytes,
          freeMemoryBytes: device.free_memory_bytes,
          mappings: []
        };
        current.mappings.push({
          binary_id: inventory.binary_id,
          logical_device_name: device.logical_device_name,
          device_id: stableId
        });
        grouped.set(stableId, current);
      }
    }
    return [...grouped.values()].sort((left, right) =>
      left.label.localeCompare(right.label)
    );
  }, [deviceInventories]);
  const workloadExperiment = experiments.find(
    (item) => item.id === workloadExperimentId
  );
  const candidateById = useMemo(
    () =>
      new Map(
        candidateOptions.map((item) => [item.candidate.id, item.candidate])
      ),
    [candidateOptions]
  );

  function logicalDevicesForInstance(instance: InstanceDraft): string[] | null {
    const discovered = discoveredDevices
      .filter((device) => selectedDevices.includes(device.id))
      .flatMap((device) =>
        device.mappings
          .filter((mapping) => mapping.binary_id === instance.binaryId)
          .map((mapping) => mapping.logical_device_name)
      );
    const unique = [...new Set(discovered)];
    if (unique.length > 0) return unique;

    const base = candidateById.get(instance.candidateId)?.candidate.placement
      .constraints.devices;
    return Array.isArray(base) && base.length > 0 ? base : null;
  }

  const unresolvedAutoDevices = instances.some(
    (instance) => logicalDevicesForInstance(instance) === null
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
    !unresolvedAutoDevices &&
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

  async function discoverDevices() {
    const binaryIds = [...new Set(instances.map((item) => item.binaryId).filter(Boolean))];
    if (binaryIds.length === 0) return;
    try {
      setDiscoveringDevices(true);
      setError(null);
      const inventories = await Promise.all(
        binaryIds.map((binaryId) => api.binaryDevices(binaryId))
      );
      const next = Object.fromEntries(
        inventories.map((inventory) => [inventory.binary_id, inventory])
      );
      setDeviceInventories(next);
      const discovered = inventories.flatMap((inventory) =>
        inventory.items.map(
          (item) => item.physical_device_key ?? item.logical_device_name
        )
      );
      const unique = [...new Set(discovered)];
      setSelectedDevices((current) => (current.length > 0 ? current : unique));
    } catch (reason) {
      setError(reason);
    } finally {
      setDiscoveringDevices(false);
    }
  }

  function toggleDevice(deviceId: string) {
    setSelectedDevices((current) =>
      current.includes(deviceId)
        ? current.filter((item) => item !== deviceId)
        : [...current, deviceId]
    );
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
              devices: logicalDevicesForInstance(item),
              n_gpu_layers: null,
              split_mode: null,
              main_gpu: null,
              tensor_split: null,
              override_tensor: []
            },
            server_identity: item.serverIdentity.trim()
          })),
          resource_policy: {
            device_memory_margin_bytes: Object.fromEntries(
              selectedDevices.map((deviceId) => [
                deviceId,
                Math.round(
                  Math.max(0, Number(deviceMarginsMiB[deviceId] || 0)) *
                    1024 *
                    1024
                )
              ])
            ),
            logical_device_mappings: discoveredDevices
              .filter((device) => selectedDevices.includes(device.id))
              .flatMap((device) => device.mappings),
            host_ram_margin_bytes: Math.round(hostRamMarginMiB * 1024 * 1024),
            allow_cpu_offload: allowCpuOffload,
            allow_swap: allowSwap,
            allowed_devices: selectedDevices,
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
                        <span>{"Instance name " + (index + 1)}</span>
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
                        <span>{"Role " + (index + 1)}</span>
                        <input
                          value={instance.role}
                          onChange={(event) =>
                            updateInstance(index, { role: event.target.value })
                          }
                        />
                      </label>
                    </div>
                    <label className="field">
                      <span>{"Base Candidate " + (index + 1)}</span>
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
                        <span>{"Model artifact " + (index + 1)}</span>
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
                        <span>{"llama-server binary " + (index + 1)}</span>
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
              <div className="section-heading-row">
                <div>
                  <h2>Host resource policy</h2>
                  <p className="section-copy">
                    Discover the exact logical devices exposed by the selected
                    llama-server binaries, then constrain the deployment to stable
                    physical GPUs and reserve per-device memory headroom.
                  </p>
                </div>
                <button
                  className="button"
                  type="button"
                  disabled={discoveringDevices || incompleteInstance}
                  onClick={() => void discoverDevices()}
                >
                  {discoveringDevices ? "Discovering…" : "Discover devices"}
                </button>
              </div>
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
              {unresolvedAutoDevices ? (
                <div className="banner banner-error">
                  At least one selected base Candidate uses automatic device
                  placement. Discover devices and keep at least one compatible
                  physical GPU selected so joint planning receives explicit logical
                  device names.
                </div>
              ) : null}
              {discoveredDevices.length > 0 ? (
                <div className="device-policy-grid">
                  {discoveredDevices.map((device) => {
                    const selected = selectedDevices.includes(device.id);
                    return (
                      <div className="device-policy-card" key={device.id}>
                        <label className="phase-option device-select">
                          <input
                            type="checkbox"
                            checked={selected}
                            onChange={() => toggleDevice(device.id)}
                          />
                          <span>{device.label}</span>
                        </label>
                        <div className="muted">
                          <code>{device.id}</code>
                        </div>
                        <div className="device-memory-line">
                          <span>
                            Free{" "}
                            {device.freeMemoryBytes === null
                              ? "—"
                              : (
                                  device.freeMemoryBytes /
                                  1024 /
                                  1024 /
                                  1024
                                ).toFixed(1) + " GiB"}
                          </span>
                          <span>
                            Total{" "}
                            {device.totalMemoryBytes === null
                              ? "—"
                              : (
                                  device.totalMemoryBytes /
                                  1024 /
                                  1024 /
                                  1024
                                ).toFixed(1) + " GiB"}
                          </span>
                        </div>
                        <label className="field">
                          <span>Reserve margin (MiB)</span>
                          <input
                            type="number"
                            min="0"
                            disabled={!selected}
                            value={deviceMarginsMiB[device.id] ?? "0"}
                            onChange={(event) =>
                              setDeviceMarginsMiB((current) => ({
                                ...current,
                                [device.id]: event.target.value
                              }))
                            }
                          />
                        </label>
                        <div className="muted">
                          {device.mappings
                            .map(
                              (mapping) =>
                                mapping.logical_device_name +
                                " via " +
                                binaries.find(
                                  (binary) => binary.id === mapping.binary_id
                                )?.path
                            )
                            .join(" · ")}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="muted resource-policy-hint">
                  Device allowlists are optional. Discover devices to bind this
                  deployment to stable physical GPU identities and reserve VRAM.
                </div>
              )}
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
