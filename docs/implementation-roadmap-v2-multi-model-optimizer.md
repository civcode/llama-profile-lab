# llama-profile-lab — V2 Multi-Model / Multi-GPU Optimization Implementation Roadmap

Status: Implementation roadmap  
Repository: civcode/llama-profile-lab  
Companion specification: docs/technical-spec-v2-multi-model-optimizer.md  
Base system: V1 single-model experiment engine

## 1. Purpose

This roadmap converts the V2 multi-model / multi-GPU optimization specification into an implementation sequence with explicit dependencies, deliverables, tests, and acceptance gates.

V2 must extend the existing V1 system rather than replace it. The implementation therefore proceeds from additive domain and persistence changes, through device/memory discovery, joint planning, simultaneous server execution, concurrent workload measurement, analysis, API/CLI/UI exposure, and finally coordinated promotion and workstation validation.

The preferred development principle is:

> Make joint placement trustworthy before making it fast, and make simultaneous measurements trustworthy before optimizing them.

Every milestone should leave the repository in a usable and testable state. V1 single-model behavior must continue to pass throughout V2 development.

## 2. Guiding implementation rules

The following rules apply throughout V2 implementation:

1. **V1 compatibility is mandatory.** Existing Candidate, placement, benchmark, server-validation, analysis, API, CLI, and UI flows remain valid.
2. **The deployment is the V2 optimization unit.** Individual V1 Candidates remain reusable building blocks.
3. **SQLite remains the source of truth.**
4. **Deployment Candidates and planned workload cases are immutable.**
5. **Completed observations are append-only.**
6. **Failures, rejected candidates, and pruning reasons are persisted.**
7. **Per-device memory is first-class data.**
8. **Memory estimates and runtime telemetry are distinct observations.**
9. **Fit proves projected feasibility, not performance.**
10. **Concurrent benchmarking is required to measure shared-resource behavior.**
11. **Aggregate throughput never replaces per-instance throughput.**
12. **Backend/device support is discovered from exact registered binaries.**
13. **Mixed-vendor operation is measured rather than assumed.**
14. **llama.cpp command-line and helper details remain behind adapters.**
15. **The optimizer does not hide a universal weighted score.**
16. **Search pruning must be deterministic and explainable.**
17. **Runtime OOM, output corruption, and telemetry gaps remain data.**
18. **Every milestone includes unit/integration tests before the next milestone begins.**
19. **Python development and CI continue to use the committed uv.lock in frozen mode.**
20. **Frontend development continues to use the committed package-lock.json.**
21. **Any GitHub Actions workflow created for V2 must use only `workflow_dispatch` unless automatic execution is explicitly requested.**
22. **Implementation work should be checkpoint-committed and pushed after meaningful subtasks so progress survives interrupted sessions.**

## 3. Milestone overview

~~~text
V2-M1  Deployment domain model + additive persistence
   ↓
V2-M2  Device inventory + per-device memory estimator
   ↓
V2-M3  Mixed-vendor composite GPU telemetry
   ↓
V2-M4  Joint deployment planner + memory feasibility pruning
   ↓
V2-M5  Simultaneous multi-server execution
   ↓
V2-M6  Concurrent DD / PP / PD / DP workloads
   ↓
V2-M7  Deployment analysis + Pareto optimization
   ↓
V2-M8  CLI + HTTP API + SSE
   ↓
V2-M9  Browser deployment workflow
   ↓
V2-M10 Coordinated promotion, hardening, and workstation acceptance
~~~

A milestone is complete only when its acceptance gate passes.

## 4. Cross-cutting workstreams

Several concerns span all milestones and should not be postponed to hardening.

### 4.1 Schema compatibility

All database changes are additive migrations. Every migration must be tested both from:

- a fresh empty database; and
- a representative V1 database upgraded in place.

No V2 migration may require rewriting historical V1 Candidate or run rows.

### 4.2 Content-addressed identity

Every new immutable object receives deterministic canonical hashing. Hash inputs must include all fields affecting:

- memory feasibility;
- placement;
- execution;
- performance;
- correctness;
- deployment-wide constraints.

Tests must prove stable hashing across serialization round trips and reject accidental dependence on map insertion order.

### 4.3 Exact binary identity

Every executable used by V2 remains registered and SHA-256 fingerprinted. This includes any new memory-estimation helper.

Capabilities are tied to the exact binary hash. A helper changing on disk after registration must fail closed until re-inspected.

### 4.4 Raw evidence retention

Persist raw JSON, stdout, stderr, argv, and relevant environment alongside normalized records wherever an external tool is invoked.

Normalized fields are convenience indexes. Raw evidence remains available for future parser changes.

### 4.5 Cancellation and cleanup

Every long-running V2 operation must observe the existing cancellation model and must clean up subprocess groups.

Tests must cover cancellation while:

- estimating placement;
- starting multiple servers;
- waiting for readiness;
- running concurrent workloads;
- stopping a partially failed deployment.

### 4.6 Deterministic testing

Hardware-independent tests use synthetic helpers, fake subprocesses, deterministic telemetry providers, and fixture outputs.

Real-GPU acceptance belongs to V2-M10, not ordinary unit tests.

---

## 5. V2-M1 — Deployment domain model and additive persistence

**Status: Implemented — targeted validation complete; full manual CI gate pending**

### Objective

Introduce the immutable deployment vocabulary and database records without changing V1 execution behavior.

This milestone establishes the data model used by every subsequent V2 feature.

### Deliverables

Domain models for:

- DeploymentCandidate;
- ModelInstanceCandidate;
- HostResourcePolicy;
- deployment workload mix/reference;
- DeploymentPlacement identity/provenance;
- device allocation/memory records;
- deployment run identity/status;
- deployment run members.

Persistence for:

- deployment_candidate;
- deployment_instance;
- deployment_placement;
- deployment_instance_placement;
- placement_device_memory;
- deployment_run;
- deployment_run_member;
- deployment rejection/pruning reason if not represented generically.

Repository methods for immutable creation and typed retrieval.

### Domain decisions

#### DeploymentCandidate

The deployment Candidate should reference existing V1 Candidate IDs rather than duplicating the complete Candidate schema.

Recommended shape:

~~~text
DeploymentCandidate
  schema = "llama-profile-deployment-candidate"
  version = 1
  instances[]
  resource_policy
  workload_mix
~~~

#### ModelInstanceCandidate

Recommended minimum fields:

~~~text
instance_id
candidate_id
role
binary_id
model_artifact_id
requested_placement
server_identity
~~~

The exact final field layout should favor normalized references where the referenced object already exists durably.

#### Stable instance identity

Instance IDs must be explicit. Array order must not be the semantic identity of an instance.

Canonical serialization should sort instances by stable instance ID before hashing if order does not have execution meaning.

### Work items

- [x] Add deployment domain module.
- [x] Add Pydantic models and validation rules.
- [x] Require at least two instances for V2 deployment optimization.
- [x] Reject duplicate instance IDs.
- [x] Validate that referenced V1 Candidates exist at persistence/service boundaries.
- [x] Define HostResourcePolicy fields.
- [x] Define deployment status/failure enums.
- [x] Add canonical deployment hashing.
- [x] Add additive migrations for deployment tables and rejection history.
- [x] Add persistence record dataclasses.
- [x] Add repositories.
- [x] Add model-to-record and record-to-model conversion helpers.
- [x] Confirm archive snapshots include V2 tables automatically through whole-database backup; no explicit enumeration change is required.
- [x] Confirm database diagnostics count all SQLite tables dynamically, including V2 tables.
- [x] Document migration semantics.

### Tests

Unit tests:

- deployment validation;
- duplicate instance rejection;
- deterministic hashing;
- hash changes for performance-relevant fields;
- stable serialization;
- HostResourcePolicy validation.

Persistence tests:

- fresh schema creation;
- V1-to-V2 migration upgrade;
- deployment round trip;
- foreign-key enforcement;
- immutable duplicate insertion behavior;
- deployment/run-member relationships.

Regression tests:

- all V1 domain and migration tests remain green;
- existing V1 database fixtures remain readable.

### Acceptance gate

The milestone passes when:

~~~bash
uv run --frozen ruff check .
uv run --frozen mypy src
uv run --frozen pytest
~~~

succeeds and tests demonstrate that a two-instance DeploymentCandidate can be persisted, reloaded, and content-addressed without modifying any V1 row.

Implementation validation completed in this environment:

- deployment/Pydantic identity tests pass in a targeted local harness;
- migrations 006 and 007 pass SQLite integrity and foreign-key checks;
- a two-instance deployment placement/rejection/run-member lifecycle passes a targeted SQLite harness.

The repository-wide Ruff, mypy, and pytest gate remains pending because the available checkout environment cannot resolve GitHub and the repository CI workflow is intentionally manual-only. It has not been dispatched.

### Suggested checkpoint commits

1. `domain: add deployment candidate models`
2. `db: add deployment persistence schema`
3. `test: cover deployment identity and migration compatibility`

---

## 6. V2-M2 — Device inventory and per-device memory estimator

**Status: Implemented — local M2 and V1/M1 regression tests passing; full repository lint/type/full-test gate pending**

### Objective

Make accelerator identity and projected per-device model/context/compute memory observable before implementing joint placement search.

This is the key feasibility primitive for the optimizer.

### Deliverables

- stable accelerator inventory;
- llama.cpp device-list adapter;
- registered memory-estimation helper;
- structured per-device memory result;
- standalone estimate persistence in memory_estimate/memory_estimate_device, with V2-M4 materialization into placement_device_memory;
- exact helper binary registration/capability detection;
- estimator cache identity;
- CLI/debug inspection path for estimator output.

### Device identity

Persist the strongest available identifiers:

~~~text
logical_device_name
backend
pci_bus_id
uuid
vendor
product_name
total_memory_bytes
driver/runtime metadata
~~~

A durable V2 device key should prefer PCI identity or hardware UUID over transient backend index.

The system must preserve both:

- durable physical identity; and
- llama.cpp logical device name used to construct argv.

### llama.cpp inventory adapter

Add an adapter around the exact registered binary's device discovery command.

Responsibilities:

- execute device listing;
- parse logical device names;
- retain raw output;
- map logical devices to host hardware identities where possible;
- expose an explicit "unresolved mapping" state rather than guessing.

### Memory-estimation helper

Implement a small native helper or supported extension of the fit tool that exposes structured JSON around llama.cpp's per-device memory APIs.

Preferred upstream primitive:

~~~text
common_get_device_memory_data(...)
~~~

Normalized output:

~~~json
{
  "devices": [
    {
      "logical_device": "CUDA0",
      "model_bytes": 0,
      "context_bytes": 0,
      "compute_bytes": 0,
      "total_bytes": 0,
      "free_bytes": 0
    }
  ],
  "resolved": {
    "n_gpu_layers": 0,
    "tensor_split": [],
    "override_tensor": []
  }
}
~~~

The concrete helper schema should be versioned.

### Helper implementation strategy

Prefer a helper that is built from the same llama.cpp source/build configuration as the target binaries.

The lab must register the helper executable independently because:

- different builds may expose different devices;
- memory behavior can change with llama.cpp revisions;
- backend compilation flags materially affect placement.

### Work items

- [x] Define accelerator/device domain type.
- [x] Extend host fingerprint with accelerator identities.
- [x] Add llama.cpp device-list parsing.
- [x] Add binary capability for structured memory estimation.
- [x] Implement helper executable contract.
- [x] Add helper JSON parser.
- [x] Add estimator adapter.
- [x] Capture model/context/compute bytes independently.
- [x] Persist device totals/free bytes at estimate time.
- [x] Persist exact selected-device ordering.
- [x] Persist raw helper output.
- [x] Add estimator cache key.
- [x] Detect helper hash drift before execution.
- [x] Add CLI inspection command for one Candidate/placement estimate.
- [x] Add fixtures covering one GPU, two homogeneous GPUs, and two heterogeneous logical devices.
- [x] Add malformed/partial JSON failure cases.

### Tests

Unit tests:

- device-list parser;
- physical/logical identity mapping;
- estimator JSON normalization;
- missing device mapping;
- negative/impossible memory values rejected;
- cache key changes with context/KV/batch/device order.

Integration tests with synthetic executables:

- helper registration;
- SHA drift rejection;
- stdout/stderr persistence;
- estimator success;
- estimator parser failure;
- nonzero exit;
- timeout/cancel.

### Acceptance gate

Given a synthetic two-device system, the service can estimate one Candidate and persist, per device:

- model bytes;
- context bytes;
- compute bytes;
- total/free device bytes;
- resolved placement arguments;
- raw helper evidence.

No throughput benchmark is required yet.

Implementation validation completed locally without GitHub Actions:

- V2-M2 domain, device-list, estimator adapter, persistence, execution, migration, and failure-path tests pass;
- V1 domain and V2-M1 regression tests remain green;
- 64 relevant pytest tests pass after the host-scoped cache and binary-drift hardening changes;
- Python compile checks pass;
- synthetic helper coverage includes one GPU, homogeneous two-GPU, and heterogeneous two-device inventories, successful estimates, cache reuse, malformed JSON, nonzero exit, timeout, cancellation, and helper SHA drift.

The V2-M2 standalone estimate tables intentionally precede deployment placement. V2-M4 will copy selected estimate rows into placement_device_memory once a concrete joint DeploymentPlacement exists; M2 does not manufacture a deployment identity merely to store a single-Candidate estimate.

The repository-wide Ruff, mypy, and complete pytest gate remains pending because Ruff/mypy are not installed or cached in the local execution environment and a complete checkout cannot be cloned through the container DNS. The manual GitHub Actions workflow has not been dispatched.

### Suggested checkpoint commits

1. `domain: add stable accelerator identities`
2. `llama: add device inventory adapter`
3. `llama: add structured memory estimator adapter`
4. `db: persist per-device placement memory`
5. `test: cover memory estimator failure and cache semantics`

---

## 7. V2-M3 — Mixed-vendor composite GPU telemetry

**Status: Implemented — local mixed-vendor telemetry and V1/M1/M2 regression tests passing; full repository lint/type/full-test gate pending**

### Objective

Observe NVIDIA and AMD/non-NVIDIA accelerators simultaneously so runtime validation can measure every device participating in a deployment.

### Existing limitation

The current AutoGpuTelemetryProvider prefers NVIDIA telemetry and returns it immediately when available. On a mixed NVIDIA/AMD host, successful nvidia-smi sampling therefore hides DRM/sysfs samples for the AMD device.

V2 replaces fallback selection with provider composition and de-duplication.

### Deliverables

- CompositeGpuTelemetryProvider;
- NVIDIA + DRM/sysfs simultaneous sampling;
- stable correlation/de-duplication;
- per-device telemetry summaries;
- preserved aggregate summary compatibility;
- mixed-vendor synthetic tests.

### Correlation strategy

Preferred matching order:

1. normalized PCI bus identity;
2. hardware UUID where comparable;
3. explicitly discovered device mapping;
4. no merge when identity is uncertain.

Never merge devices based only on product name.

### Merge policy

When NVIDIA and sysfs report the same physical NVIDIA GPU:

- prefer nvidia-smi fields where present;
- fill missing values from sysfs;
- retain source metadata.

Non-NVIDIA sysfs devices remain in the sample.

### Per-device summary metrics

Add normalized metrics such as:

~~~text
telemetry.gpu.<stable_device_key>.utilization_avg_pct
telemetry.gpu.<stable_device_key>.utilization_peak_pct
telemetry.gpu.<stable_device_key>.vram_used_peak_bytes
telemetry.gpu.<stable_device_key>.temperature_peak_c
telemetry.gpu.<stable_device_key>.power_avg_w
telemetry.gpu.<stable_device_key>.power_peak_w
~~~

If metric-key constraints make a raw hardware key unsuitable, introduce a stable encoded device identifier and expose the display name separately.

### Work items

- [x] Introduce composite GPU provider.
- [x] Query NVIDIA and sysfs providers independently.
- [x] Add PCI identity normalization.
- [x] Add record merge/de-duplication.
- [x] Retain provider/source provenance.
- [x] Keep existing aggregate summary metrics.
- [x] Add per-device summary structure.
- [x] Emit per-device generic metrics.
- [x] Update telemetry API DTOs if required.
- [x] Update frontend telemetry types only if backend DTO shape changes.
- [x] Add mixed NVIDIA/AMD fixture tree.
- [x] Test partial provider failure.
- [x] Test duplicate NVIDIA device correlation.
- [x] Test two unrelated GPUs with same/similar names are not merged.

### Run-quality policy

Do not classify intentional utilization from the other deployment instance as "external GPU load" during concurrent deployment runs.

V2 deployment runs need deployment-aware quality classification:

- before/after utilization can still indicate unrelated contamination;
- during-run utilization of participating GPUs is expected;
- all member processes belong to the same benchmark activity.

Add tests before concurrent execution depends on this behavior.

### Acceptance gate

A deterministic telemetry test with one synthetic NVIDIA device and one synthetic AMD DRM device returns both devices in every sample and emits distinct per-device summaries.

All existing V1 telemetry tests remain green.

Implementation validation completed locally without GitHub Actions:

- one synthetic NVIDIA device plus one synthetic AMD/sysfs device are returned together;
- duplicate NVIDIA/sysfs observations merge by normalized PCI identity, with NVIDIA values preferred and sysfs filling missing fields;
- explicit stable mappings can correlate devices when PCI/UUID is unavailable;
- unrelated devices with identical product names remain distinct;
- partial provider failure preserves telemetry from surviving providers;
- provider provenance is retained on raw samples and per-device summaries;
- aggregate V1 telemetry metrics remain present while per-device generic metrics are emitted under stable encoded IDs;
- intentional during-run GPU utilization does not become external-GPU contamination when before/after baselines are quiet;
- 81 relevant pytest tests pass across V1 domain, V2-M1, V2-M2, and V2-M3 coverage;
- Python compile checks pass.

No schema migration is required because raw GPU observations are already persisted as JSON and generic metric rows already support the new per-device metric names. The API telemetry DTO reuses the domain TelemetrySample type, so the new fields are additive. The current frontend has no dedicated telemetry sample type requiring a synchronized TypeScript change.

The repository-wide Ruff, mypy, and complete pytest gate remains pending because Ruff/mypy are not installed or cached in the local execution environment and the full repository cannot be cloned through the container DNS. The manual GitHub Actions workflow has not been dispatched.

### Suggested checkpoint commits

1. `telemetry: compose GPU providers across vendors`
2. `telemetry: add stable per-device summaries`
3. `test: cover mixed NVIDIA AMD telemetry`

---

## 8. V2-M4 — Joint deployment planner and memory feasibility pruning

**Status: Implemented — local M4 core and V1/M1/M2/M3 regression tests passing; full repository lint/type/full-test gate pending**

### Objective

Generate deployment-wide placement candidates, estimate all instance allocations, combine their device memory, and reject impossible configurations before throughput benchmarking.

### Deliverables

- deployment search-space representation;
- joint placement expansion;
- device-allocation policy;
- memory feasibility service;
- capability pruning;
- deterministic rejection reasons;
- planner preview counts;
- persisted planned deployment cases.

### Planner inputs

A deployment experiment should contain:

~~~text
base deployment candidate
search dimensions
search constraints
deployment workload suite
measurement policy
placement policy
~~~

Search dimensions may address instance-specific V1 Candidate paths and deployment placement fields.

Examples:

~~~text
instances.qwen.context.size
instances.qwen.context.cache_type_k
instances.qwen.context.cache_type_v
instances.qwen.requested_placement.tensor_split
instances.flash.requested_placement.tensor_split
resource_policy.device_memory_margin_bytes.CUDA0
~~~

The concrete path syntax should be deterministic and validated by a registry.

### Search phases

Planner expansion should occur in this order:

1. expand raw deployment dimensions;
2. validate V1 Candidate relationships;
3. apply explicit user constraints;
4. validate binary/device capabilities;
5. run/lookup per-instance memory estimates;
6. aggregate projected memory per physical device;
7. apply margins and host constraints;
8. reject infeasible candidates;
9. persist valid deployment Candidates/cases.

### Memory equation

For every physical device:

~~~text
projected_device_bytes =
    Σ model_bytes(instance, device)
  + Σ context_bytes(instance, device)
  + Σ compute_bytes(instance, device)
  + reserved_deployment_overhead_bytes
~~~

Feasible only when:

~~~text
projected_device_bytes + margin_bytes <= usable_device_bytes
~~~

### Avoid order-biased fitting

Do not:

1. fit instance A against all currently free VRAM;
2. then fit B against the remainder.

Instead, the requested joint resource allocation must be explicit before estimation. Each instance estimator receives constraints/margins derived from the deployment candidate.

### Initial search strategy

The first planner remains deterministic rather than Bayesian/adaptive.

Support:

- discrete tensor/layer split ratios;
- device subsets;
- context sizes;
- KV types;
- batch/ubatch;
- memory margins.

Use constraints to bound candidate count.

### Rejection reasons

Persist normalized reasons such as:

~~~text
unsupported_device
unsupported_backend_pair
unsupported_split_mode
invalid_kv_configuration
memory_estimate_failed
device_memory_exceeded
host_memory_policy_violation
candidate_constraint_failed
~~~

Include structured details, especially required versus available bytes.

### Work items

- [x] Add deployment search-space models.
- [x] Add instance-addressable parameter registry.
- [x] Add planner expansion.
- [x] Add capability pruning.
- [x] Add memory estimate lookup/execution.
- [x] Add per-device aggregation.
- [x] Add margin checks.
- [x] Add pruning persistence.
- [x] Add planner preview service.
- [x] Add deterministic case hashing.
- [x] Add symmetry-reduction hooks.
- [x] Avoid symmetry reduction for heterogeneous devices by default.
- [x] Add planner CLI preview.
- [x] Add tests for raw/valid/rejected counts.

### Tests

Required planner fixtures:

1. one-model-per-GPU baseline;
2. both models split across both GPUs;
3. exact-fit candidate;
4. candidate exceeding GPU0 only;
5. candidate exceeding GPU1 only;
6. unsupported device pair;
7. invalid split mode;
8. high-context candidate rescued by lower-precision KV;
9. high-context candidate rejected even after allowed placement options;
10. deterministic duplicate candidate elimination.

### Acceptance gate

A synthetic two-model/two-GPU deployment can be planned and produces:

- stable raw candidate count;
- stable capability-rejected count;
- stable memory-rejected count;
- stable valid count;
- persisted rejection details;
- no throughput process launches.

Implementation validation completed locally without GitHub Actions:

- deployment dimensions address instance Candidate fields, requested placement fields, device subsets, tensor splits, KV types, contexts, batch/ubatch, and per-device margins;
- explicit logical-to-physical mappings are available when strong hardware identity cannot be discovered, and unresolved mappings fail closed rather than guessing;
- explicit deployment placement constraints are passed into the M2 estimator without mutating the V1 Candidate;
- helper-resolved device order, split mode, main GPU, explicit tensor split, tensor overrides, and explicit numeric GPU-layer requests are validated against the joint planner request before a result can be persisted;
- capability pruning rejects unsupported devices, backend pairs, split modes, tensor/KV combinations, and missing exact-binary options before throughput work;
- memory feasibility aggregates model/context/compute bytes across every instance on each physical device and applies per-device margins against usable memory;
- feasible cases materialize estimator evidence into DeploymentPlacement/placement_device_memory while rejected cases persist normalized reasons and required-versus-available byte details;
- one-model-per-GPU, both-models-split-both-GPUs, exact-fit, GPU0 overflow, GPU1 overflow, backend-pair rejection, invalid split, lower-precision-KV rescue, all-placement high-context rejection, duplicate elimination, and estimator failure fixtures are covered;
- 100 locally executable pytest tests pass across the reconstructed V1/V2-M1/M2/M3/M4 core workspace;
- Python compile checks pass;
- the branch also contains CLI preview/plan parser and dispatch tests; they are not included in the 99-test local count because the reconstructed workspace does not contain the full V1 CLI/API/analysis module tree.

The planner never launches llama-bench, SPEED-Bench, or llama-server throughput workloads. It only uses exact registered inventories/capabilities plus the M2 memory-estimator path. Preview does not persist deployment plan cases or rejection history; plan persists deterministic feasible cases and explainable rejection history. Symmetry reduction is opt-in, so heterogeneous devices are never assumed interchangeable by default.

The repository-wide Ruff, mypy, and complete pytest gate remains pending because Ruff/mypy are not installed or cached in the local execution environment and the full repository cannot be cloned through the container DNS. The manual GitHub Actions workflow has not been dispatched.

### Suggested checkpoint commits

1. `planning: add deployment search dimensions`
2. `planning: add joint memory feasibility pruning`
3. `db: persist deployment planning rejections`
4. `cli: preview deployment plans`
5. `test: cover joint placement planner`

---

## 9. V2-M5 — Simultaneous multi-server execution

**Status: Implemented — synthetic real-process lifecycle coverage passing locally; full repository lint/type/full-test gate pending**

### Objective

Launch and manage all model instances of one DeploymentCandidate at the same time under a single deployment lifecycle.

This milestone proves residency and process orchestration before adding concurrent performance workloads.

### Deliverables

- DeploymentExecutor;
- one managed llama-server per instance;
- deterministic per-instance ports/endpoints;
- readiness barrier;
- deployment-wide cancellation;
- complete process-group cleanup;
- runtime memory-headroom validation;
- deployment server logs/status persistence.

### Execution sequence

~~~text
acquire host lock
  ↓
load deployment placement
  ↓
build argv for every instance
  ↓
start all servers
  ↓
wait for all readiness checks
  ↓
capture runtime telemetry/memory
  ↓
mark deployment ready
  ↓
optional residency hold/probe
  ↓
shutdown all servers
  ↓
release lock
~~~

If any instance fails to start, every already-started instance must be terminated.

### Port allocation

Ports must be collision-safe and deterministic within one deployment run.

Prefer explicit executor-managed free-port reservation rather than hard-coded offsets that may collide with external services.

Persist the actual endpoint used by every member run.

### Server argv

Reuse the existing llama-server adapter where possible.

V2 should provide each instance with:

- its V1 Candidate;
- resolved placement;
- selected model path;
- selected binary;
- instance-specific port;
- any deployment-required environment isolation.

Do not fork a second copy of V1 argv logic.

### Runtime memory validation

After all servers are ready, capture GPU telemetry and/or llama.cpp memory breakdown evidence.

Compare observed VRAM against planned margins.

A configured hard margin violation becomes a persisted deployment failure or invalidation according to policy.

### Failure semantics

Required cases:

- first server fails;
- later server fails after earlier server is ready;
- readiness timeout;
- binary drift;
- model artifact missing;
- runtime OOM;
- cancellation during startup;
- cleanup failure;
- stale deployment run after process loss.

### Work items

- [x] Add DeploymentExecutor.
- [x] Reuse host lock.
- [x] Add multi-server process registry.
- [x] Add per-instance server state.
- [x] Add readiness barrier.
- [x] Add port allocator.
- [x] Add deployment status transitions.
- [x] Add deployment cancellation.
- [x] Add cleanup on partial startup.
- [x] Add runtime memory validation.
- [x] Add server log persistence.
- [x] Add stale-run recovery.
- [x] Add residency smoke probe.
- [x] Add execution service tests.

### Tests

Integration tests with fake servers must verify:

- both servers become ready;
- one failure tears down both;
- cancellation tears down both;
- readiness order does not affect final status;
- duplicate ports are never assigned;
- process groups are terminated;
- host lock excludes other benchmark activity;
- runtime margin violations persist correctly.

### Acceptance gate

A synthetic deployment starts two independent fake llama-server processes, reaches a single deployment-ready state, persists both member identities/endpoints, and shuts both down cleanly.

No concurrent benchmark clients are required yet.

Implementation validation completed locally without GitHub Actions:

- two independent executable fake llama-server processes reach one deployment-ready barrier, receive distinct reserved ports, persist endpoints/PIDs/argv/readiness/logs, and shut down cleanly;
- readiness completion order does not affect the final deployment status;
- first-member failure, later-member failure, readiness timeout, cancellation, binary drift, missing model artifact, runtime OOM, cleanup failure, leader-process crash with a surviving child, host-lock contention, and stale-run recovery are covered;
- process-group cleanup terminates child processes even when the server leader has already exited;
- runtime headroom validation compares observed free VRAM with the M4 reserved margin and persists runtime_memory_margin_violated when the hard margin is breached;
- missing planned-device GPU telemetry now fails closed as telemetry_incomplete instead of silently accepting an unvalidated margin;
- schema version 10 persists explicit deployment member endpoint/status/PID/argv/model paths/readiness/exit/log/forced-kill/cleanup evidence;
- llprof deployment execute accepts a persisted placement plus per-instance model paths and performs the residency lifecycle without starting M6 benchmark clients;
- the pushed M5 executor suite contains 14 real-process lifecycle/failure tests, with deployment CLI coverage alongside it;
- the reconstructed V1/V2-M1/M2/M3/M4 regression workspace remains green at 100 pytest tests after the M5 persistence/executor changes;
- Python compile checks pass.

The repository-wide Ruff, mypy, complete pytest suite, and frontend gates remain pending because the local reconstruction is intentionally partial. The manual GitHub Actions workflow has not been dispatched.

### Suggested checkpoint commits

1. `execution: add deployment server lifecycle`
2. `execution: add readiness barrier and cleanup`
3. `execution: validate runtime memory margins`
4. `test: cover partial multi-server failures`

---

## 10. V2-M6 — Concurrent DD / PP / PD / DP workloads

**Status: Implemented — deterministic concurrent-workload and production-client validation passing locally; full repository lint/type/full-test gate pending**

### Objective

Measure real shared-resource performance while all model instances remain resident and active.

### Deliverables

- concurrent workload domain representation;
- synchronized workload client orchestration;
- DD phase;
- PP phase;
- PD phase;
- DP phase;
- overlap-aware aggregate throughput;
- standalone/concurrent retention metrics;
- deployment-level quality classification;
- raw member result persistence.

### Workload phases

For two model instances A and B:

| Phase | A | B | Primary question |
| --- | --- | --- | --- |
| DD | decode | decode | How much aggregate generation capacity is available? |
| PP | prefill | prefill | How much aggregate prompt-processing capacity is available? |
| PD | prefill | decode | How much does a large prompt disrupt interactive generation? |
| DP | decode | prefill | What happens in the inverse mixed phase? |

The underlying workload records must remain generic enough to extend beyond exactly two instances later.

### Synchronization

Use an explicit start barrier.

Clients should be initialized before the barrier where possible so setup time does not distort overlap.

Record:

~~~text
client_ready_ns
barrier_release_ns
first_request_ns
first_token_ns
last_token_ns
finished_ns
~~~

Use a monotonic clock for interval calculations.

### Aggregate throughput

Do not add independent rates.

For an overlap interval:

~~~text
combined_tps =
  sum(tokens completed during overlap) / overlap_duration
~~~

Persist both:

- member-native throughput;
- overlap-normalized aggregate throughput.

Where exact token event timestamps are unavailable, define and document a conservative interval method and retain enough raw timing to improve it later.

### Standalone baseline

Retention requires a comparable standalone result.

The analysis layer should find the exact or nearest valid standalone baseline only under explicitly matching dimensions such as:

- Candidate;
- placement;
- workload depth;
- batch settings;
- binary;
- host fingerprint.

If no valid baseline exists, retention is unavailable rather than guessed.

### Retention metrics

~~~text
instance_retention = concurrent_tps / standalone_tps
throughput_loss_pct = (1 - retention) * 100
min_retention = min(all instance retentions)
~~~

For latency-sensitive workloads also persist latency change.

### Context asymmetry

Required workload support:

- both instances shallow;
- both instances deep;
- Qwen deep / Flash shallow;
- Qwen shallow / Flash deep where meaningful.

This is necessary because KV occupancy changes contention and feasible placement.

### Correctness hooks

Concurrent clients must preserve existing output-validation facilities.

A successful HTTP response with high TPS is not sufficient when output validation fails.

### Work items

- [x] Add concurrent workload models.
- [x] Add DD/PP/PD/DP planner generation.
- [x] Add synchronized client barrier.
- [x] Add shared timing record.
- [x] Add overlap interval calculation.
- [x] Add aggregate token accounting.
- [x] Persist per-member raw results.
- [x] Persist per-member normalized metrics.
- [x] Add standalone-baseline lookup.
- [x] Add retention metrics.
- [x] Add deployment-aware quality classification.
- [x] Add correctness validation hooks.
- [x] Add asymmetric-context cases.
- [x] Add timeout and member-failure behavior.

### Tests

Deterministic fake-client tests must cover:

- exact simultaneous start;
- staggered completion;
- no-overlap rejection;
- partial overlap;
- one failed member;
- one timed-out member;
- correct aggregate TPS arithmetic;
- retention lookup ambiguity;
- missing standalone baseline;
- output-validation failure;
- cancellation during concurrent phase.

### Acceptance gate

A deterministic two-instance synthetic run executes DD, PP, PD, and DP and persists:

- per-instance throughput;
- aggregate overlap-aware throughput;
- standalone reference;
- retention;
- minimum retention;
- timing evidence;
- quality;
- failure/correctness status.

Implementation notes and validation:

- the existing Candidate-dependent workload-suite expander supplies concrete prefill/decode depth cases, so equal shallow/deep and asymmetric-context combinations are generated without a second context model;
- `DeploymentExecutor` exposes a typed resident-action seam, allowing all M6 phases to run under the same M5 host lock while the same server processes remain resident;
- clients are prepared before a shared `threading.Barrier`; `client_ready_ns`, `barrier_release_ns`, request/token/finish timestamps, and cumulative token events use the monotonic clock;
- the production client prewarms `depth_tokens`, then streams llama-server `/completion` with prompt-progress and predicted-token counters;
- aggregate PP/TG rates are computed from cumulative tokens completed inside the common overlap interval, never by summing independent member rates;
- raw member persistence includes normalized token-event evidence for every client implementation, not only the production HTTP client;
- standalone baselines match Candidate, resolved placement, host, binary, workload mode, prompt/generate counts, and depth exactly; identical imports are idempotent while conflicting exact matches are surfaced as `baseline_ambiguous`;
- per-member throughput retention, throughput loss, baseline latency, concurrent latency, latency increase, and deployment minimum retention are persisted when an unambiguous baseline exists;
- missing baselines remain `baseline_missing` rather than being guessed;
- member failure, timeout, cancellation, no-overlap, and output-validation failure retain phase/member evidence and fail the deployment run with normalized failure semantics;
- schema migrations 011 and 012 add concurrent workload/result persistence and latency-retention fields; the current schema version is 12;
- `llprof deployment benchmark SPEC` launches the M5 residency lifecycle and executes the generated concurrent phase matrix, with optional exact standalone baselines supplied in the benchmark spec;
- deterministic tests cover exact synchronized start, partial/staggered overlap, no overlap, aggregate arithmetic, missing/ambiguous baselines, latency deltas, member failure, member timeout, correctness failure, cancellation, asymmetric depths, and all DD/PP/PD/DP phases;
- production-client protocol tests cover context-depth prewarm, prompt-progress token accounting, raw event evidence, and decode token-count mismatch rejection;
- a local reconstructed pure-M6 validation pass completed with all focused planner/overlap/HTTP-client tests green;
- migrations 011 and 012 apply cleanly in an executable SQLite harness; integrity and foreign-key checks pass and representative concurrent phase/member/baseline latency rows persist successfully;
- changed M6 Python/test files pass the branch-side line-length/trailing-whitespace/blank-run hygiene scan.

The full repository Ruff, mypy, complete pytest suite, frontend gates, and real llama.cpp two-model workstation acceptance remain pending because the execution environment cannot clone GitHub over DNS. The manual GitHub Actions workflow has not been dispatched.

### Suggested checkpoint commits

1. `domain: add concurrent deployment workloads`
2. `execution: synchronize deployment workload clients`
3. `analysis: compute overlap aggregate throughput`
4. `analysis: add standalone retention metrics`
5. `test: cover DD PP PD DP orchestration`

---

## 11. V2-M7 — Deployment analysis and Pareto optimization

**Status: Implemented — targeted branch-side validation complete; full repository lint/type/full-test gate pending**

### Objective

Make joint-placement results explorable without hiding per-model tradeoffs.

### Deliverables

- deployment result service;
- memory-matrix projection;
- interference analysis;
- baseline comparison;
- deployment Pareto frontiers;
- context-capacity analysis;
- CSV/JSON export extensions.

### Analysis dimensions

Support projections over:

- model instance;
- device;
- split ratio;
- GPU layers;
- context size;
- KV K/V type;
- batch/ubatch;
- workload phase;
- prompt depth;
- decode depth;
- binary/backend configuration.

As in V1, ambiguous hidden dimensions must not be silently averaged.

### Core metrics

Required metrics include:

~~~text
deployment.combined_tg_tps
deployment.combined_pp_tps
deployment.min_retention
deployment.total_validated_context_tokens
deployment.device.<id>.headroom_bytes
deployment.total_power_avg_w
deployment.total_power_peak_w
~~~

Retain instance metrics alongside them.

### Memory matrix

Add a projection where rows are instance/category and columns are devices:

~~~text
                    CUDA0      Vulkan0
qwen.model
qwen.context
qwen.compute
flash.model
flash.context
flash.compute
reserved
projected_free
runtime_peak
~~~

Projected and runtime values should be visually/semantically distinguishable.

### Pareto service

Reuse the V1 objective model where possible.

Example objectives:

~~~text
combined_dd_tps:max
combined_pp_tps:max
min_retention:max
qwen_context_tokens:max
min_device_headroom_bytes:max
total_power_avg_w:min
~~~

Constraints are applied before Pareto calculation.

### Domination rules

A failed, correctness-invalid, or constraint-invalid deployment is excluded from the valid Pareto set but remains queryable.

Do not drop failure rows from raw export.

### Work items

- [x] Add deployment analysis repository queries.
- [x] Add deployment metric namespace.
- [x] Add memory matrix projection.
- [x] Add interference/retention view.
- [x] Add deployment baseline comparison.
- [x] Extend Pareto service to deployment candidates.
- [x] Add context-capacity metrics.
- [x] Add power aggregation.
- [x] Add CSV/JSON export.
- [x] Add exact-filter/facet handling.
- [x] Add ambiguity rejection tests.
- [x] Add failed-candidate visibility.

### Tests

- Pareto dominance with three or more objectives;
- constraint filtering before Pareto;
- failed/corrupt candidates excluded from valid frontier;
- exact hidden-dimension rejection;
- memory projection correctness;
- projected/runtime distinction;
- export round trip;
- baseline delta sign conventions.

### Acceptance gate

Given a fixture database containing multiple joint placements, analysis can identify a non-dominated set using combined decode TPS, combined PP TPS, minimum retention, context capacity, and device headroom without losing per-instance metrics.

Implementation notes and validation:

- `DeploymentAnalysisService` treats a persisted deployment placement as the optimization point and repeated deployment runs as evidence for that point;
- exact analysis coordinates include instance Candidate settings, resolved placement fields, tensor-split elements, binary IDs, persisted accelerator backends, workload phase/member depth, and per-device projected allocation fields;
- rate/latency projections reject hidden workload-coordinate ambiguity rather than averaging across prompt/decode depths;
- the deployment metric namespace includes combined PP/TG TPS, minimum retention, total validated context, runtime/projected device headroom, total runtime power, and dynamic per-instance throughput/retention/latency metrics;
- unqualified headroom metrics are runtime-only and require complete evidence for every planned device; projected headroom remains a separate explicit namespace so Pareto vectors do not mix projected and runtime semantics;
- memory matrices separate projected model/context/compute/reserved/free rows from runtime peak-used and minimum-free rows;
- interference views preserve aggregate phase metrics plus per-instance native/overlap throughput, standalone references, retention/loss, latency deltas, and correctness;
- placement comparison uses candidate-minus-baseline signs and percentage deltas against the absolute baseline magnitude;
- deployment Pareto analysis applies explicit metric constraints before dominance and excludes placements with no valid completed/correctness-valid concurrent evidence while retaining their raw rows for export;
- six-objective fixture coverage exercises DD throughput, PP throughput, minimum retention, validated context capacity, runtime headroom, and total average power;
- JSON/CSV export preserves successful, failed, and correctness-invalid deployment observations with per-instance configuration and result fields;
- migration 013 adds timestamped GPU samples across the resident deployment interval, enabling runtime peak-memory and sample-weighted average/peak total-power metrics; schema version is 13;
- the M5 residency executor now samples GPUs from readiness through resident action/probe and persists those snapshots without changing the existing manual-only workflow policy;
- the M7 fixture tests cover projected/runtime memory distinction, per-instance interference, backend coordinates, hidden-dimension rejection, signed baseline deltas, pre-Pareto constraints, correctness-invalid/failed exclusion, multi-objective dominance, runtime power aggregation, and export round-trip;
- an executable SQLite check confirmed the additive deployment GPU telemetry table, foreign-key behavior, and integrity constraints.

The container still cannot resolve GitHub, so the full repository Ruff, mypy, complete pytest suite, frontend gates, and real workstation acceptance remain pending. The GitHub Actions workflow has not been dispatched.

### Suggested checkpoint commits

1. `analysis: add deployment metrics and projections`
2. `analysis: add deployment interference comparisons`
3. `analysis: extend Pareto service to deployments`
4. `test: cover deployment analysis ambiguity and dominance`

---

## 12. V2-M8 — CLI, HTTP API, and SSE

**Status: Implemented (full local acceptance suite pending)**

### Objective

Expose deployment planning, execution, inspection, and analysis through the same service boundary principles as V1.

### Deliverables

CLI commands:

~~~text
llprof deployment create
llprof deployment plan
llprof deployment placement
llprof deployment run
llprof deployment pause
llprof deployment resume
llprof deployment cancel
llprof deployment show
llprof deployment results
llprof deployment pareto
~~~

HTTP resources:

~~~text
POST /api/deployments
POST /api/deployments/{id}/plan
POST /api/deployments/{id}/run
POST /api/deployments/{id}/pause
POST /api/deployments/{id}/resume
POST /api/deployments/{id}/cancel

GET /api/deployments/{id}
GET /api/deployments/{id}/candidates
GET /api/deployments/{id}/placements
GET /api/deployments/{id}/runs
GET /api/deployments/{id}/results
GET /api/deployments/{id}/pareto
~~~

SSE events for deployment state.

### DTO design

Do not expose database record shapes directly.

Typed DTOs should include:

- deployment definition;
- planning summary;
- rejection summary;
- placement memory matrix;
- run/member status;
- current throughput where meaningful;
- final analysis results.

### Progress model

A deployment progress snapshot should report:

~~~text
deployment_status
planned_candidates
completed_candidates
failed_candidates
active_deployment_run
member_states[]
current_workload_phase
~~~

### CLI rendering

Human-readable CLI should clearly distinguish:

- projected versus runtime memory;
- per-instance versus aggregate TPS;
- valid versus rejected candidates;
- placement IDs versus deployment IDs.

### Work items

- [x] Add application service methods.
- [x] Add API DTOs.
- [x] Add API routes.
- [x] Add error mapping.
- [x] Add SSE deployment snapshots.
- [x] Add CLI parser tree.
- [x] Add CLI rendering.
- [x] Add JSON-friendly output where existing CLI convention supports it.
- [x] Reuse V1 result/analysis services where semantics match.
- [x] Add API/CLI tests.
- [x] Verify V1 routes unchanged.

### Tests

API:

- create/plan/get;
- invalid instance reference;
- plan summary;
- run/cancel/resume;
- placement memory response;
- Pareto response;
- SSE snapshot changes;
- member failure propagation.

CLI:

- help text;
- planning preview;
- placement matrix;
- results;
- Pareto;
- failure exit behavior.

### Acceptance gate

The complete V2 workflow through M7 can be driven without importing internal Python modules, using only CLI or HTTP APIs.

All V1 CLI/API tests remain green.

Implementation notes and validation:

- deployment operations are durable in SQLite, so pause/cancel requests can be issued by a separate API or CLI process and resume can reconstruct the persisted execution request;
- HTTP and CLI share the same deployment planning, execution, memory, results, and Pareto services instead of duplicating M4-M7 semantics;
- deployment progress/SSE snapshots retain active and terminal member state, current Candidate/placement identity, the latest workload phase, aggregate PP/TG throughput, runtime/projected memory evidence, and failure details where available;
- progress avoids reusing prior-run runtime metrics while a newly started operation has not yet created its own deployment run;
- API acceptance coverage exercises create/plan/get, invalid references, plan summaries, run/pause/resume/cancel, placement memory, results, Pareto, SSE changes, failed member propagation, and preservation of the V1 route surface;
- CLI acceptance coverage exercises help, create/show/placement, preview/plan, results, Pareto, durable control dispatch, and nonzero failure behavior;
- schema 14 persists durable deployment operation state and request payloads used for cross-process control/resume;
- GitHub Actions were not dispatched. The current environment cannot clone/resolve GitHub for a full local Ruff, mypy, pytest, and frontend run, so the repository-wide acceptance suite remains explicitly unverified here.

### Suggested checkpoint commits

1. `api: expose deployment planning and inspection`
2. `api: expose deployment execution and SSE`
3. `cli: add deployment workflow`
4. `test: cover deployment API and CLI`

---

## 13. V2-M9 — Browser deployment workflow

**Status: Implemented (production frontend gate pending)**

### Objective

Provide an interactive workflow for authoring, running, and comparing multi-model placements without hiding the underlying resource tradeoffs.

### Deliverables

- deployment creation page;
- model-instance editor;
- device/resource policy editor;
- planning preview;
- rejection explanation UI;
- placement memory matrix;
- execution progress;
- DD/PP/PD/DP results;
- interference view;
- deployment Pareto view;
- candidate detail page.

### UI sequence

~~~text
Select base Candidates
  ↓
Assign instance roles
  ↓
Select allowed devices
  ↓
Define context / KV / placement dimensions
  ↓
Define memory margins and constraints
  ↓
Preview plan
  ↓
Inspect pruning
  ↓
Run
  ↓
Inspect memory + telemetry
  ↓
Compare concurrency phases
  ↓
Select Pareto finalists
~~~

### Deployment editor

The editor should avoid requiring users to manually type internal IDs.

Use existing Candidate/launcher data to populate choices.

For each instance expose:

- model identity;
- base Candidate;
- context/KV settings;
- placement dimensions;
- device set;
- server settings.

Deployment-level controls expose:

- margins;
- CPU offload policy;
- swap policy;
- minimum retention;
- workload mix.

### Placement memory matrix

This is a primary V2 visualization, not a debug page.

Required distinctions:

- weights/model;
- context/KV;
- compute;
- reserved margin;
- projected free;
- runtime peak where available.

Flag infeasible cells before execution.

### Interference view

Show, per instance:

~~~text
standalone TPS
concurrent TPS
retention %
latency delta
~~~

and deployment aggregates beside them.

Avoid a visualization that shows only combined TPS.

### Pareto view

Reuse V1 Pareto interaction patterns where possible, adding deployment objectives.

Allow filtering by:

- phase;
- context;
- KV type;
- device placement;
- correctness status;
- minimum retention.

### Work items

- [x] Add frontend deployment types.
- [x] Add API client calls.
- [x] Add deployment routes/pages.
- [x] Add instance editor.
- [x] Add resource-policy editor.
- [x] Add plan preview.
- [x] Add pruning reason view.
- [x] Add placement memory matrix.
- [x] Add run progress/SSE integration.
- [x] Add concurrency results table/charts.
- [x] Add interference visualization.
- [x] Add Pareto visualization.
- [x] Add candidate detail.
- [x] Add accessibility/keyboard review.
- [x] Add component/unit tests.
- [ ] Run production build/typecheck/test gate in a local checkout.

### Tests

Frontend unit/component tests for:

- editing two instances;
- device selection;
- invalid duplicate instance IDs;
- planning preview counts;
- rejection details;
- memory matrix rendering;
- projected/runtime labels;
- DD/PP/PD/DP rendering;
- per-instance retention;
- Pareto selection;
- failure states.

### Acceptance gate

From the browser, a user can define a two-model deployment, preview memory feasibility, run the experiment, inspect simultaneous performance, and identify Pareto finalists without using the CLI.

Required frontend gates:

~~~bash
cd frontend
npm ci
npm run typecheck
npm run test
npm run build
~~~

Implementation notes and validation:

- deployment authoring uses persisted Candidates, model artifacts, exact llama-server binaries, and workload suites rather than requiring raw internal IDs as the primary workflow;
- exact-binary device discovery is exposed through the existing inventory/correlation service so the editor can select stable physical GPUs, persist logical-to-physical mappings, and reserve per-device memory margins;
- planning supports validated instance Candidate/placement dimensions, deployment memory margins, conditions, constraints, non-persisting preview, and persisted feasible cases/rejections;
- deployment detail uses SSE plus durable controls for run/pause/resume/cancel, keeps projected/runtime memory evidence distinct, renders canonical DD/PP/PD/DP observations, and links generated Candidates to detailed pruning/interference evidence;
- deployment Candidate details show standalone versus overlap TPS, retention/loss, latency deltas, correctness state, rejection explanations, and feasible placement memory;
- deployment Pareto supports deployment objectives, phase and exact-coordinate filters, minimum-retention constraints, a frontier table, and scatter visualization;
- component coverage includes authoring, duplicate IDs, exact-device selection, memory labels, phase/failure rendering, rejection/interference evidence, Pareto filters/scatter, routing, and accessibility labels;
- GitHub Actions were not dispatched. The current environment cannot obtain a runnable local checkout, so `npm ci`, `npm run typecheck`, `npm run test`, and `npm run build` remain explicitly pending.

### Suggested checkpoint commits

1. `frontend: add deployment editor and planning preview`
2. `frontend: add placement memory matrix`
3. `frontend: add concurrent results and interference views`
4. `frontend: add deployment Pareto workflow`
5. `test: cover deployment browser workflow`

---

## 14. V2-M10 — Coordinated promotion, hardening, and workstation acceptance

**Status: Implementation complete; automated clean-checkout and target-workstation acceptance pending**

### Objective

Close the V2 experiment lifecycle, prove migration/failure safety, and validate the optimizer on the target heterogeneous workstation.

### Deliverables

- coordinated deployment promotion proposal;
- V2 archive/export completeness;
- corruption/failure hardening;
- full migration coverage;
- target-workstation acceptance;
- V2 known-limitations documentation;
- release checklist/update.

### Coordinated promotion

A V2 promotion proposal must include all affected model profiles together.

Required provenance:

- source launcher snapshot;
- every source model profile;
- every proposed model profile;
- exact model artifacts;
- binary/helper hashes;
- selected devices;
- placement args;
- context/KV settings;
- memory projection;
- runtime memory evidence;
- concurrent validation results;
- correctness results;
- unified reviewable patch.

Promotion must refuse a partial proposal when the selected deployment depends on coordinated resource allocation.

### Hardening cases

Add tests for:

- malformed memory-helper JSON;
- helper binary changes after registration;
- device inventory changes after planning;
- GPU disappears between plan and run;
- runtime VRAM exceeds projected allocation;
- simultaneous startup OOM;
- only one server survives startup;
- process cleanup failure;
- stale deployment state after crash;
- output corruption;
- missing standalone retention baseline;
- telemetry provider partial failure;
- archive restore with V2 tables;
- historical V1 database upgrade;
- launcher source drift before promotion.

### Documentation

Add/update:

- getting-started V2 deployment section;
- benchmark workflow V2;
- architecture V2;
- troubleshooting V2;
- known limitations;
- release checklist;
- README status.

Do not rewrite V1 documentation as if V2 had always existed; preserve historical V1 documents where useful.

## 14.1 Target-workstation acceptance plan

The target acceptance system includes:

- NVIDIA RTX 4070 Ti Super with 16 GiB VRAM;
- AMD GPU discovered and identified by the actual llama.cpp build;
- Qwen3.8 27B target configuration;
- Flash Next resident simultaneously.

Exact backend choice is discovered from working registered builds and is not predetermined by the roadmap.

### Phase A — hardware and binary inventory

Verify:

- both physical accelerators are detected;
- stable PCI/device identities are persisted;
- chosen llama.cpp binary/binaries expose the intended devices;
- helper memory estimator sees the same device ordering;
- NVIDIA and AMD telemetry appear simultaneously.

Record exact:

- GPU names;
- PCI IDs;
- driver versions;
- backend names;
- llama.cpp commit/build identity;
- binary SHA-256 values.

### Phase B — single-model memory truth

For Qwen and Flash independently:

- run memory estimator;
- launch server;
- compare projected versus runtime VRAM;
- record error/headroom;
- repeat across at least two context/KV configurations.

Acceptance requires estimates accurate enough for safe pruning with the configured margin.

If estimator/runtime deltas are systematic, persist and document an overhead policy rather than silently increasing margins.

### Phase C — baseline topology

Benchmark the intuitive baseline:

~~~text
Qwen -> one GPU
Flash -> the other GPU
~~~

Run:

- standalone PP/TG;
- simultaneous DD;
- simultaneous PP;
- PD;
- DP.

This baseline is mandatory even if the optimizer later prefers another topology.

### Phase D — split-placement topology

Evaluate at least one feasible topology where both model instances consume memory on both GPUs.

For Qwen, include a configuration that materially increases context and/or KV precision relative to the single-4070-fit configuration when feasible.

Verify:

- exact per-device weight memory;
- exact per-device context memory;
- runtime headroom;
- no unexpected CPU spill unless explicitly allowed;
- no swap;
- output correctness.

### Phase E — search experiment

Run a bounded search including:

- at least three Qwen placement allocations;
- at least three Flash placement allocations or appropriate constrained alternatives;
- at least two Qwen KV precisions when supported;
- multiple Qwen context targets;
- one-model-per-GPU baseline;
- both-models-split candidates.

The exact grid should be small enough for repeatable acceptance but large enough to demonstrate pruning and Pareto selection.

### Phase F — finalist repetitions

For the non-dominated finalists:

- at least three repetitions per canonical phase;
- representative deep-context occupancy;
- thermal stabilization/warmup policy;
- clean telemetry requirement;
- correctness validation.

### Phase G — acceptance assertions

V2 workstation acceptance passes when:

1. both GPUs are visible with stable identities;
2. mixed-vendor telemetry captures both at once;
3. per-instance/per-device model/context/compute memory is persisted;
4. joint feasibility pruning rejects known-overcommitted candidates;
5. both servers remain resident for all canonical workload phases;
6. DD/PP/PD/DP produce per-instance and aggregate metrics;
7. standalone-to-concurrent retention is available;
8. the one-model-per-GPU baseline is present;
9. at least one both-models-on-both-GPUs candidate is present;
10. Pareto analysis returns valid finalists;
11. correctness checks pass on promoted finalists;
12. no hidden swap or unintended CPU fallback occurs;
13. promotion generates one coordinated deployment proposal;
14. all automated backend and frontend tests remain green.

### Work items

- [x] Add coordinated promotion model.
- [x] Add deployment proposal diff.
- [x] Extend deployment export/archive.
- [x] Extend archive restore tests.
- [x] Add V2 database-check coverage.
- [x] Add hardening fixtures.
- [x] Add device-change revalidation.
- [x] Add runtime projection-delta reporting.
- [x] Add correctness probes for finalists.
- [x] Write V2 operator docs.
- [x] Add machine-checkable workstation acceptance report.
- [ ] Execute target-workstation acceptance.
- [ ] Record acceptance artifacts/results.
- [ ] Update known limitations from measured behavior.
- [x] Prepare draft release notes/checklist.

### Automated acceptance gate

Before workstation validation:

~~~bash
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen mypy src
uv run --frozen pytest

cd frontend
npm ci
npm run typecheck
npm run test
npm run build
~~~

Any GitHub Actions workflow used to run these checks must remain manually triggered with:

~~~yaml
on:
  workflow_dispatch:
~~~

unless automatic execution is explicitly requested.

Implementation notes and validation:

- schema 15 persists append-only coordinated deployment promotion proposals with source/proposed launcher snapshots, per-instance source mappings, unified changes, placement identity, and validation evidence;
- promotion refuses partial source coverage, launcher source drift, missing completed/correctness-valid configured phases, and missing persisted memory-estimator provenance; proposal generation never mutates launcher configuration;
- complete V2 provenance export is available through `llprof deployment export`, covering immutable deployment/Candidate data, plans/rejections, placements/memory, estimator/device evidence, raw deployment/workload evidence, GPU telemetry, durable operations, environment identities, and promotion proposals;
- archive restore coverage now proves schema-15 coordinated promotion evidence survives a hashed SQLite archive round trip;
- `llprof database check` includes representative indexed deployment plan, placement-run, and promotion queries;
- deployment execution re-discovers exact-binary accelerator inventory before server startup and fails as `device_capability_mismatch` when a planned logical selector disappears or remaps to a different physical GPU;
- deployment memory analysis reports signed runtime-versus-projection peak-use and free-headroom deltas per physical device;
- the production concurrent client classifies malformed/non-object SSE, regressing token counters, contradictory final counters, and exact token-count mismatches as correctness-invalid output;
- V2 operator documentation now includes getting started, architecture, benchmark workflow, troubleshooting, known limitations, release checklist, and draft release notes;
- `llprof deployment acceptance-report` evaluates persisted target-workstation evidence without launching new workloads and explicitly leaves Pareto selection, target identity confirmation, and clean-checkout gates pending for operator/release sign-off;
- GitHub Actions were not dispatched. This environment still cannot obtain a runnable clean checkout, so the locked Python and frontend acceptance gates remain explicitly pending.

### Final acceptance gate

V2 is complete only after both:

1. all automated gates pass from a clean checkout; and
2. the target-workstation acceptance assertions above are recorded against the intended Qwen and Flash model artifacts and exact llama.cpp builds.

### Suggested checkpoint commits

1. `promotion: add coordinated deployment proposals`
2. `archive: include V2 deployment provenance`
3. `test: harden deployment failure paths`
4. `docs: add V2 operations and troubleshooting`
5. `docs: record workstation acceptance results`

---

## 15. Recommended implementation order inside milestones

The milestone boundaries are hard dependencies, but work inside a milestone should also follow a small-step sequence.

### Backend change pattern

For each new backend concept:

1. domain type;
2. validation tests;
3. persistence migration;
4. repository tests;
5. service;
6. adapter/executor integration;
7. CLI/API surface;
8. documentation.

This minimizes the amount of behavior built on untested persistence assumptions.

### External process pattern

For each new helper/server/client integration:

1. argv builder;
2. capability validation;
3. parser;
4. fixture tests;
5. subprocess wrapper integration;
6. persistence of raw output;
7. cancellation/timeout handling;
8. service-level integration.

### Frontend change pattern

For each new API surface:

1. TypeScript DTO;
2. API client method;
3. isolated component;
4. component test;
5. page integration;
6. route/state integration;
7. production build.

## 16. Test matrix

The V2 test suite should cover the following dimensions without requiring real GPUs.

| Area | Minimum synthetic cases |
| --- | --- |
| Device inventory | NVIDIA only, AMD only, mixed, ambiguous identity |
| Memory estimator | 1 GPU, 2 GPU, malformed output, timeout, binary drift |
| Planner | feasible, GPU0 OOM, GPU1 OOM, unsupported backend, exact fit |
| Telemetry | NVIDIA, sysfs, mixed, duplicate merge, provider failure |
| Multi-server | both ready, first fails, second fails, timeout, cancel |
| Workloads | DD, PP, PD, DP, partial overlap, member failure |
| Analysis | aggregate TPS, retention, Pareto, hidden dimension ambiguity |
| Promotion | coordinated diff, source drift, invalid finalist |
| Migration | fresh V2, V1 upgrade, archive restore |
| UI | author, preview, run, results, interference, Pareto |

## 17. Performance of the optimizer itself

The experiment planner must remain practical as dimensions increase.

Instrument at least:

- raw candidate expansion time;
- estimator cache hit rate;
- number of estimator executions;
- number of memory-pruned candidates;
- number of benchmarked candidates;
- total planning time.

Avoid running the same per-instance memory estimate repeatedly when the estimator identity is unchanged.

A later optimization may memoize partial deployment sums, but correctness comes first.

## 18. Search-space growth controls

Provide explicit mechanisms to keep early experiments bounded:

- discrete placement ratios;
- min/max context;
- allowed KV types;
- allowed device sets;
- minimum VRAM margin;
- minimum expected retention after initial measurements;
- finalist count limit for expensive validation.

Do not silently truncate a requested search. If a configured planner limit is introduced, planning must report that limit and which candidates were omitted.

## 19. Observability and diagnostics

Every deployment run should make it possible to answer:

- Which exact models and binaries ran?
- Which devices were visible?
- Which devices did each model use?
- How many bytes of weights/context/compute were projected on each device?
- How much VRAM was actually used?
- Which servers were running?
- Which workload phase was active?
- What was each model's throughput?
- What was the aggregate throughput?
- How much did each model slow down relative to standalone?
- Did thermal/power/noise conditions contaminate the run?
- Did correctness validation pass?
- Why was a candidate rejected or failed?

If the stored data cannot answer one of those questions, the milestone producing that data is not complete.

## 20. Documentation checkpoints

Documentation should evolve with implementation rather than waiting until V2-M10.

After each milestone:

- update the V2 technical spec if implementation resolves an open design question;
- mark implemented roadmap work items;
- add operator-facing documentation for commands that now exist;
- update known limitations for discovered backend restrictions.

Avoid claiming target-workstation behavior before it has been measured.

## 21. Branch and commit strategy

The canonical V2 development branch for this repository is:

~~~text
feature/multi-model-gpu-optimizer-spec
~~~

Commit V2 implementation checkpoints directly to that branch. Do not use `main` as the routine V2 development target, and do not create auxiliary PR, staging, or milestone branches unless explicitly requested.

The branch should retain checkpoint commits for each milestone:

~~~text
feature/multi-model-gpu-optimizer-spec
  checkpoint commits for V2-M1
  checkpoint commits for V2-M2
  ...
~~~

Commit after each meaningful subtask and push promptly.

Do not defer all work into one final commit.

Before risky or lengthy activities such as:

- native llama.cpp helper compilation;
- full Python test suite;
- frontend production build;
- target-workstation benchmark campaign;

create and push a checkpoint commit.

## 22. Definition of V2 complete

V2 is complete when the system can take a set of simultaneously resident models and a heterogeneous multi-GPU host and answer, with persisted evidence:

> Which joint placement and runtime configuration gives the best acceptable combination of aggregate prompt processing, aggregate decode throughput, per-model service retention, context capacity, memory headroom, correctness, and power?

The answer must be based on:

- exact binary/model identities;
- measured or normalized per-device memory;
- joint feasibility;
- simultaneous workload execution;
- per-model and aggregate performance;
- runtime telemetry;
- explicit user-selected Pareto objectives and constraints.

A configuration is not considered optimal merely because it fits, and a configuration is not considered valid merely because it is fast.
