# Getting started with V2 multi-model deployments

V2 extends the V1 experiment system with joint placement, simultaneous residency, synchronized DD/PP/PD/DP workloads, deployment Pareto analysis, and coordinated review-only promotion.

V1 Candidates remain the per-model source of truth. A V2 DeploymentCandidate combines two or more persisted Candidates with model artifacts, exact llama-server binaries, resource policy, and a shared workload suite.

## Prerequisites

Before authoring a deployment:

1. Create and plan the V1 experiments that provide the source Candidates.
2. Register the exact llama-server and llama-memory-estimator executables you intend to use.
3. Register the model artifacts referenced by the deployment.
4. Keep launcher configuration available when you want to generate a coordinated promotion proposal.

The exact executable identities matter. Binary SHA-256 drift, device inventory drift, or source launcher-profile drift fails closed.

## Browser workflow

Start the local UI as usual and open **Deployments**.

### 1. Create the base deployment

Choose at least two persisted Candidates. For each instance select:

- a stable instance name and role;
- the source Candidate;
- model artifact;
- exact llama-server binary.

The resource-policy editor can discover the selected binaries' current accelerator inventory. Select the physical GPUs the deployment is allowed to use and set per-device memory reserves. Logical-to-physical mappings are persisted so mixed backends can refer to the same physical GPU without relying on product names.

Choose the workload source experiment and canonical phases. DD, PP, PD, and DP are enabled by default.

### 2. Preview and persist a joint plan

On the deployment page, choose one llama-memory-estimator binary per instance and add search dimensions.

Supported instance paths include Candidate context/KV/compute fields and deployment placement fields such as:

```text
instances.<id>.context.size
instances.<id>.context.cache_type_k
instances.<id>.context.cache_type_v
instances.<id>.requested_placement.devices
instances.<id>.requested_placement.n_gpu_layers
instances.<id>.requested_placement.tensor_split
resource_policy.device_memory_margin_bytes.<physical-device-id>
```

Values are entered as JSON arrays, so a device-set dimension can contain arrays:

```json
[["CUDA0"], ["CUDA0", "Vulkan0"]]
```

Use **Preview plan** first. Preview performs the same capability, device, estimator, and memory-feasibility checks as planning but does not persist plan cases. Inspect constraint, capability, estimator, and memory rejection counts, then use **Persist plan**.

### 3. Run a feasible placement

Select a planned placement and start the deployment. Before any server process starts, execution re-hashes the exact server binaries and re-discovers their device inventories. Logical selectors must still resolve to the physical GPUs planned earlier; a missing/remapped device is persisted as `device_capability_mismatch`.

All server instances must become ready. Runtime GPU headroom is checked against the planned memory margin. Missing planned-device telemetry fails closed.

The browser receives durable progress through SSE and supports pause, resume, and cancel. Control state is persisted in SQLite, so another CLI/API process can issue the control request.

### 4. Inspect simultaneous performance

The deployment page keeps projected and runtime memory evidence separate. Per-device memory cards also report signed deltas:

- observed peak used minus projected allocation;
- observed minimum free minus projected free headroom.

Positive peak-use error or negative free-headroom error means runtime evidence was worse than the projection.

Use generated Candidate detail pages for:

- persisted pruning explanations;
- exact generation coordinates;
- per-instance standalone and concurrent TPS;
- retention and throughput loss;
- latency and latency delta;
- correctness status;
- placement memory.

Raw deployment results remain available even for failed or correctness-invalid rows. Invalid evidence is excluded from the valid Pareto frontier.

### 5. Select Pareto finalists

Choose deployment objectives such as combined DD decode TPS, PP throughput, minimum retention, validated context, runtime headroom, or power. Exact phase/coordinate filters and minimum-retention constraints apply before dominance.

The UI shows both a frontier table and a two-objective scatter view.

### 6. Generate a coordinated promotion proposal

On a finalist Candidate page, select its finalized placement and the source V1 experiment for every instance.

Promotion is review-only. It refuses:

- partial instance coverage;
- launcher source drift;
- a placement without a completed joint run;
- missing correctness-valid configured phases;
- missing memory-estimator provenance.

The proposal contains every affected profile, exact binary/helper hashes, model artifacts, resolved placement arguments, projected/runtime memory evidence, concurrent validation evidence, and one unified launcher diff. The launcher file is never modified by proposal generation.

### 7. Export and archive

Export the complete V2 workflow provenance:

```bash
llprof deployment export DEPLOYMENT_ID \
  --output deployment-provenance.json \
  --database data/benchmarks.db
```

This export is different from `llprof deployment results`: it includes immutable definitions, planning/rejections, placements, estimator evidence, raw run/member/workload evidence, GPU telemetry, durable operations, environment records, and promotion proposals.

Take a full SQLite archive for restore/recovery with the existing `llprof archive` command. Schema 15 archives retain all V2 tables and coordinated promotion records.

See also:

- `docs/benchmark-workflow-v2.md`
- `docs/architecture-v2.md`
- `docs/troubleshooting-v2.md`
- `docs/known-limitations-v2.md`
- `docs/release-checklist-v2.md`
