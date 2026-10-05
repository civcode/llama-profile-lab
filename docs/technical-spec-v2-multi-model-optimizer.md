# V2 Technical Specification: Multi-Model / Multi-GPU Optimization

Status: Draft  
Target: post-V1 extension  
Scope: local single-host deployments with multiple simultaneously resident llama.cpp model instances

## 1. Purpose

V1 optimizes the configuration of one model Candidate at a time and resolves placement for that Candidate independently. V2 extends the optimization unit from one Candidate to an entire deployment containing multiple model instances that share the same CPU, RAM, GPUs, PCIe fabric, power envelope, and background load.

The primary target is a workstation where two or more models must remain loaded at the same time and where assigning one complete model to each GPU is not necessarily optimal. A representative configuration is:

- one NVIDIA GPU and one AMD GPU;
- a large dense model such as Qwen3.8 27B;
- a second latency-oriented model such as Flash Next;
- both models simultaneously resident;
- context sizes large enough that KV-cache precision materially changes memory use;
- a goal of maximizing useful aggregate prompt-processing and token-generation capacity without starving either model.

The optimizer must be able to discover solutions where each model has weights, context memory, and compute buffers on more than one device.

## 2. Goals

V2 shall:

1. Represent a deployment as an immutable optimization candidate containing multiple model instances.
2. Model GPU placement per model and per device rather than treating a GPU as owned by one model.
3. Persist measured or projected per-device memory for model weights, context/KV state, and compute buffers.
4. Reject infeasible deployment candidates before expensive benchmarking when projected device memory exceeds configured limits.
5. Benchmark all model instances while they are simultaneously resident.
6. Measure interference under decode/decode, prefill/prefill, prefill/decode, and decode/prefill concurrency.
7. Optimize aggregate prompt-processing throughput and aggregate decode throughput while retaining per-model service metrics.
8. Preserve V1's explicit Pareto-analysis philosophy rather than introducing a hidden universal score.
9. Support heterogeneous device sets when the selected llama.cpp build advertises them.
10. Record exact backend/device identities, binary fingerprints, model artifacts, placement arguments, and memory estimates needed to reproduce a result.
11. Improve GPU telemetry so NVIDIA and non-NVIDIA devices can be observed in the same sample.
12. Keep V1 single-model experiments valid and readable.

## 3. Non-goals

The first V2 implementation will not:

- become a distributed multi-host scheduler;
- assume that all combinations of CUDA, HIP, Vulkan, or other backends can cooperate in one llama.cpp process;
- assume that heterogeneous multi-GPU placement is faster than independent single-GPU placement;
- infer performance from VRAM fit alone;
- automatically change production launcher configuration without the existing review/promotion boundary;
- require tensor parallelism;
- use a single weighted score as the canonical definition of optimal;
- model arbitrary queueing systems in the initial release.

## 4. Design principles

The V1 principles remain in force, with the following additions:

- The deployment, not the model, is the resource-allocation unit.
- Fit is a feasibility operation; benchmark data decides performance.
- Simultaneously resident models must be benchmarked simultaneously when evaluating shared-resource behavior.
- Per-device memory is first-class data.
- Per-model metrics must never be lost when aggregate metrics are calculated.
- Heterogeneous backend support is capability-discovered and experimentally validated.
- Placement estimates and runtime telemetry are different evidence and are both retained.
- Search pruning must be explainable and persisted.

## 5. Existing V1 boundary

The current V1 Candidate represents one target model, one context configuration, one compute configuration, one placement configuration, one server configuration, and optional speculative-decoding configuration.

ResolvedPlacement similarly describes the concrete placement for one Candidate through fields such as:

- production context size;
- GPU layer count;
- split mode;
- main GPU;
- selected devices;
- tensor split;
- tensor overrides.

PlacementResolver resolves and caches placement independently per Candidate. This remains useful and should not be removed. V2 adds a deployment layer above these objects.

## 6. New domain model

### 6.1 DeploymentCandidate

Introduce an immutable content-addressed DeploymentCandidate.

Conceptually:

~~~text
DeploymentCandidate
  schema
  version
  host_constraints
  instances[]
  workload_mix
  resource_policy
~~~

Each deployment Candidate contains two or more ModelInstanceCandidate records.

A deployment hash must include every performance- or feasibility-relevant field. Reordering instances must not accidentally change semantics; instance identity shall therefore be explicit and deterministic.

### 6.2 ModelInstanceCandidate

A ModelInstanceCandidate references one immutable V1 Candidate and adds deployment-level identity and execution information.

Required fields:

~~~text
instance_id
candidate_id
role
model_artifact_id
binary_id
requested_placement
server_identity
~~~

The V1 Candidate remains the source of model, context, KV types, batch, ubatch, threads, and other llama.cpp parameters.

The deployment layer owns resource sharing and simultaneous execution.

### 6.3 HostResourcePolicy

HostResourcePolicy defines limits that apply to the complete deployment:

~~~text
device_memory_margin_bytes[device_id]
host_ram_margin_bytes
allow_cpu_offload
allow_swap
allowed_devices[]
allowed_backend_pairs[]
maximum_total_power_w (optional)
~~~

Memory margins are constraints, not optimization results.

### 6.4 DeploymentPlacement

DeploymentPlacement is the concrete resolved placement of every instance.

Conceptually:

~~~text
DeploymentPlacement
  deployment_candidate_id
  host_id
  instance_placements[]
  device_allocations[]
  feasibility
  provenance
~~~

Each instance placement contains the existing ResolvedPlacement fields plus per-device memory records.

## 7. Device identity and capability discovery

### 7.1 Stable device identity

The optimizer must not use an integer GPU index as the durable device identity because indices can vary between backends or boots.

Persist, when available:

- llama.cpp device name, for example CUDA0 or Vulkan0;
- backend name;
- PCI bus identity;
- vendor and product name;
- hardware UUID;
- total device memory;
- driver/runtime metadata available from host discovery.

A host hardware fingerprint shall include the set of relevant accelerator identities and topology metadata.

### 7.2 llama.cpp device discovery

The exact registered llama.cpp binary must be queried for its visible device set and supported placement arguments. Device discovery belongs with binary capability discovery and must be fingerprinted with the executable SHA-256.

The optimizer must not manufacture mixed-backend combinations that the exact binary cannot address.

### 7.3 Backend compatibility

A deployment candidate is feasible only if each model instance can be launched with its requested device set under the selected binary.

The first implementation must treat mixed NVIDIA/AMD execution as capability-dependent. It shall not encode a global rule that CUDA plus Vulkan, CUDA plus HIP, or any other pair always works.

## 8. Per-device memory accounting

### 8.1 Required memory categories

For every model instance and device, persist:

~~~text
model_bytes
context_bytes
compute_bytes
total_instance_bytes
~~~

For every physical device, also persist:

~~~text
device_total_bytes
device_free_bytes_at_measurement
reserved_margin_bytes
projected_deployment_bytes
projected_free_bytes
~~~

The optimizer must distinguish model weights from context/KV memory because increasing context size or changing KV precision affects context memory without changing weight placement.

### 8.2 Preferred llama.cpp interface

Current llama.cpp exposes common_get_device_memory_data in common/fit.h. It loads a model and context with no_alloc and returns common_device_memory_data per participating device:

~~~text
total
free
model
context
compute
~~~

This is the preferred memory-estimation primitive when the registered build exposes a suitable callable adapter.

Current llama.cpp also exposes llama_get_memory_breakdown at the lower API layer. The implementation may use either interface, but the persisted normalized result must follow this specification.

### 8.3 Adapter strategy

llama-profile-lab should not directly link its Python process against arbitrary local llama.cpp builds.

Add a small registered helper executable or extend the existing llama-fit-params integration so that the lab can request structured JSON containing:

- device identities;
- fitted placement arguments;
- per-device model/context/compute bytes;
- relevant model hyperparameters used during fit;
- exit status and diagnostics.

The helper executable must be fingerprinted like other llama.cpp tools.

Text log parsing may be supported as a compatibility fallback, but structured output is preferred.

The V2-M2 implementation resolves this boundary as a separately registered binary kind,
`llama-memory-estimator`, with a strict version-1 JSON contract documented in
`docs/memory-estimator-helper-v1.md`. Successful standalone estimates are cached by stable
host identity, Candidate semantic hash, model artifact identity, helper SHA-256, and exact
placement/device-order inputs. Attempts retain argv, stdout, stderr, exit status, duration,
and typed failure state.

Standalone estimates are stored in `memory_estimate` and `memory_estimate_device`.
`placement_device_memory` remains tied to a concrete joint `DeploymentPlacement`; the
joint planner introduced in V2-M4 materializes the selected standalone estimate rows into
that table rather than inventing deployment identity during M2.

### 8.4 Multi-model memory feasibility

For an independently served deployment, projected memory on a device is:

~~~text
projected_device_bytes =
  sum(instance.model_bytes
      + instance.context_bytes
      + instance.compute_bytes)
  + deployment_reserved_overhead_bytes
~~~

A candidate is memory-feasible only when:

~~~text
projected_device_bytes + reserved_margin_bytes
  <= usable_device_bytes
~~~

The feasibility calculation must be performed independently for every device.

Do not use common_fit_extra_model as a general substitute for this calculation. Its upstream semantics are suitable for related extra models such as draft/MTP contexts and do not represent arbitrary independent servers with unrelated context lifecycles.

### 8.5 Runtime validation

Projected fit does not prove runtime fit. Finalists must be launched simultaneously and runtime device-memory telemetry must confirm that configured margins survive initialization and representative workloads.

OOM during load or workload execution is persisted as data.

## 9. Placement search space

### 9.1 Placement dimensions

The deployment search space may include:

- selected devices per instance;
- split mode per instance;
- main GPU per instance;
- GPU layer count;
- tensor split;
- tensor overrides;
- per-device fit margins;
- context size;
- K cache type;
- V cache type;
- batch size;
- ubatch size;
- server parallelism;
- model-specific compute settings already supported by V1.

### 9.2 Layer split as initial target

Layer split is the default multi-GPU placement mode for the first V2 implementation because it is the mature llama.cpp multi-GPU path and naturally distributes layer-owned state.

Tensor split/tensor-parallel modes may be added when supported by the exact binary and model architecture, but must be capability-gated and marked experimental in analysis when upstream describes them as experimental.

### 9.3 Joint allocation

Placement search must not fit model A to all free memory and then give the remainder to model B. That introduces order bias.

Instead, the planner generates or derives a joint memory allocation policy.

For two devices and two model instances this can be represented initially by per-instance device fractions or per-device reserved margins. Example:

~~~text
GPU0:
  Qwen target share: 11.0 GiB
  Flash target share: 4.0 GiB
  safety margin: 1.0 GiB

GPU1:
  Qwen target share: 7.0 GiB
  Flash target share: 8.0 GiB
  safety margin: 1.0 GiB
~~~

The actual llama.cpp placement remains measured, not assumed from these target shares.

### 9.4 Symmetry reduction

The planner should eliminate equivalent candidates when two devices or two model instances are truly interchangeable. Heterogeneous GPUs are not interchangeable by default.

All pruning decisions must record the rule that removed the candidate.

### 9.5 V2-M4 implementation boundary

V2-M4 implements the first deterministic joint planner with instance-addressable paths such as
`instances.<id>.context.size`, `instances.<id>.requested_placement.devices`, and
`resource_policy.device_memory_margin_bytes.<device>`. Search expansion is content-addressed,
constraint-filtered, and de-duplicates equivalent effective placement overrides. Symmetry
reduction is available only through an explicit hook and is disabled by default.

Joint planning requires an explicit selected device set before estimation; `auto` placement is
rejected rather than sequentially fitting one instance and giving another the remainder.
The M2 estimator accepts deployment placement overrides without mutating the V1 Candidate.

For each expanded deployment point, the planner:

1. validates the exact registered server binary's relevant placement capability surface;
2. resolves logical devices to durable physical identities from M2 inventory or explicit policy mappings;
3. applies allowed-device/backend policy and tensor/KV compatibility rules;
4. obtains one structured M2 memory estimate per instance under the requested placement;
5. aggregates model/context/compute bytes by physical device;
6. applies per-device margins against the estimator-reported usable/free bytes;
7. persists normalized rejection reasons or one feasible DeploymentPlacement.

Migration 009 stores deployment-plan count/provenance records and deterministic feasible case
references. Successful standalone M2 estimates remain canonical; feasible M4 placements
materialize their selected rows into `placement_device_memory`. Rejected points do not launch
throughput work.

The CLI accepts a versioned-domain search document inside a small planning request file:

~~~json
{
  "base_deployment_candidate_id": "deploy_...",
  "search_space": {
    "dimensions": [
      {
        "path": "instances.qwen.context.size",
        "values": [131072, 196608]
      }
    ]
  },
  "instances": [
    {
      "instance_id": "qwen",
      "helper_binary_id": "bin_...",
      "model_path": "/models/qwen.gguf"
    },
    {
      "instance_id": "flash",
      "helper_binary_id": "bin_...",
      "model_path": "/models/flash.gguf"
    }
  ],
  "timeout_seconds": 300
}
~~~

`llprof deployment preview` evaluates stable raw/rejected/valid counts without persisting
deployment plan cases or rejection history. `llprof deployment plan` persists feasible cases
and explainable rejection history.

## 10. Search strategy

An exhaustive Cartesian product becomes expensive quickly. V2 therefore uses staged search.

### Stage A: capability and memory pruning

Generate candidate placements and reject:

- unsupported device selections;
- unsupported split modes;
- impossible KV-type/backend combinations;
- projected VRAM overcommit;
- disallowed CPU offload or swap;
- invalid V1 Candidate relationships.

This stage performs no throughput benchmark.

### Stage B: standalone characterization

Measure each relevant model configuration independently on relevant placements.

Record:

- prompt-processing throughput;
- decode throughput;
- latency;
- per-device memory;
- power and thermals.

Standalone results establish upper bounds and provide interference denominators.

### Stage C: concurrent screening

Launch all deployment instances and run a reduced concurrent workload suite. Drop candidates that are clearly dominated or unstable.

### Stage D: finalist validation

Run the complete workload mix, more repetitions, longer duration, and production-like server validation.

### Stage E: optional adaptive search

A later milestone may use prior measurements to propose new joint placements between sampled points. The initial implementation can remain deterministic grid/constraint based.

## 11. Concurrent workload model

### 11.1 Required two-model phases

For a two-instance deployment, define four canonical phases:

| Phase | Instance A | Instance B | Purpose |
| --- | --- | --- | --- |
| DD | decode | decode | steady-state shared decode capacity |
| PP | prefill | prefill | peak prompt-processing/resource pressure |
| PD | prefill | decode | prefill interference with interactive generation |
| DP | decode | prefill | inverse mixed interference |

Workloads must start close enough in time that their measured intervals overlap.

### 11.2 Synchronization

Concurrent runs require an orchestration barrier.

The executor shall:

1. launch all servers;
2. wait for all readiness checks;
3. prepare workload clients;
4. synchronize workload start;
5. record a shared deployment-run start timestamp;
6. run all clients concurrently;
7. retain per-client results independently;
8. stop measurement after all required clients finish or a deployment-level timeout occurs.

If one instance fails, the run remains a deployment failure even if another instance completes.

The V2-M6 implementation reuses the M5 resident-server lifecycle. Each generated phase prepares every client before a shared `threading.Barrier`; the barrier action records the monotonic release timestamp used by all members. The production llama-server client prewarms `depth_tokens` before the barrier, then streams `/completion` with prompt-progress and predicted-token counters. Per-member evidence retains client-ready, barrier-release, request, token, and finish timestamps plus cumulative token events.

Generated phases come from the existing Candidate-dependent workload-suite expander. For two instances, the planner forms the Cartesian product of concrete prefill/decode cases required by DD/PP/PD/DP, which naturally includes equal-depth and asymmetric-depth combinations when shallow and deep cases are present in the suite.

### 11.3 Workload depth

Prompt and decode depths remain explicit. Mixed phases should include shallow and deep-context cases because placement interference can change as KV occupancy grows.

The planner must support both equal-context tests and asymmetric tests, such as a deep Qwen context while Flash Next remains at a smaller context.

## 12. Metrics

### 12.1 Per-instance throughput

Persist existing throughput metrics per instance, namespaced by instance identity.

Examples:

~~~text
deployment.instance.qwen.pp_tps
deployment.instance.qwen.tg_tps
deployment.instance.flash.pp_tps
deployment.instance.flash.tg_tps
~~~

### 12.2 Aggregate throughput

For overlapping workload intervals:

~~~text
combined_decode_tps =
  sum(instance decode tokens) / overlap_duration

combined_prompt_processing_tps =
  sum(instance prompt tokens) / overlap_duration
~~~

Do not sum independently measured rates from non-overlapping intervals.

M6 uses cumulative prompt/decode token events to count only tokens observed inside the common active interval. The common interval starts at the latest member request start and ends at the earliest member finish. A phase with no positive common interval is persisted as `no_overlap` and is not assigned aggregate throughput.

### 12.3 Retention / interference

For each instance and workload kind:

~~~text
throughput_retention =
  concurrent_throughput / standalone_throughput
~~~

Also persist:

~~~text
throughput_loss_pct
latency_increase_pct
~~~

These metrics expose candidates that maximize aggregate throughput by starving one service.

M6 standalone baselines are exact-match evidence keyed by Candidate, resolved placement, host, binary, workload mode, prompt/generate token counts, and depth. Re-importing the exact same evidence is idempotent. Multiple conflicting exact matches are surfaced as `baseline_ambiguous`; no match is `baseline_missing`. Neither condition is guessed away. When baseline latency is available, M6 also persists the baseline latency and the concurrent percentage increase.

### 12.4 Fairness

The initial fairness metric is minimum throughput retention across deployment instances:

~~~text
min_retention = min(instance_retention)
~~~

Additional fairness functions may be added later, but raw per-instance values remain canonical.

### 12.5 Capacity metrics

Persist:

- maximum validated context per instance;
- KV cache type per instance;
- total validated context tokens across the deployment;
- device memory headroom;
- host RAM headroom;
- average and peak power where available.

## 13. Optimization objectives

V2 retains explicit Pareto objectives.

Typical objectives include:

- maximize combined decode TPS;
- maximize combined prompt-processing TPS;
- maximize minimum per-model throughput retention;
- maximize validated context capacity;
- maximize minimum VRAM headroom;
- minimize p95 request latency;
- minimize total power;
- minimize host-RAM spill.

The UI and CLI must make objective directions explicit.

A deployment that has higher aggregate throughput but violates a configured minimum retention or context requirement is infeasible, not merely lower-scoring.

## 14. Constraints

Deployment-level constraints may include:

~~~text
qwen.context.size >= 196608
qwen.context.cache_type_k in {q8_0, f16}
qwen.context.cache_type_v in {q8_0, f16}
flash.context.size >= 131072
min_retention >= 0.70
device[CUDA0].headroom_mib >= 768
device[Vulkan0].headroom_mib >= 768
host.swap_used_bytes == 0
~~~

The constraint system should reuse V1's exact filtering philosophy and fail closed on unknown dimensions.

## 15. Execution architecture

### 15.1 Independent server processes first

The initial implementation shall manage one llama-server process per model instance.

Advantages:

- independent logs;
- independent readiness;
- independent ports;
- clear PID attribution;
- independent failure status;
- easier workload synchronization;
- straightforward mapping from instance configuration to server argv.

A future milestone may validate the same deployment through llama.cpp multi-model routing when desired.

### 15.2 Deployment lock

V1 uses a host lock to prevent benchmark contamination. V2 retains one exclusive host benchmark lock for a deployment run.

All model servers and workload clients belonging to the deployment execute under that lock.

### 15.3 Process lifecycle

Deployment execution owns a process group for every server and client. Cancellation or terminal failure must clean up every process started by the deployment run.

Stale-running recovery must recognize deployment runs in addition to V1 benchmark runs.

The V2-M5 implementation launches one managed llama-server per deployment instance under the existing exclusive host lock. It reserves collision-safe local ports before startup, persists the actual endpoint and argv for every member, launches all members before entering a concurrent readiness barrier, and treats the deployment as ready only after every member reports a healthy `/health` endpoint.

Shutdown is deployment-wide. Partial startup, readiness timeout, cancellation, interruption, runtime memory failure, or member crash tears down every registered process group. Process-group termination is attempted even when the server leader has already exited so children cannot survive an otherwise terminal deployment run.

After readiness, M5 samples current per-device GPU memory and compares observed free VRAM against the M4 reserved margin. A hard margin violation is persisted as `runtime_memory_margin_violated`. If a planned physical device cannot be observed, validation fails closed as `telemetry_incomplete` instead of treating the margin as validated.

## 16. Telemetry extensions

### 16.1 Composite GPU provider

The current AutoGpuTelemetryProvider returns NVIDIA samples when nvidia-smi succeeds and otherwise falls back to DRM/sysfs. On a mixed NVIDIA/AMD host this hides the AMD GPU whenever NVIDIA telemetry is available.

Replace this fallback behavior with a composite provider:

1. query NVIDIA management telemetry when available;
2. query DRM/sysfs telemetry;
3. correlate records using PCI identity or another stable hardware key;
4. merge duplicate NVIDIA observations;
5. retain non-NVIDIA DRM devices in the same sample.

### 16.2 Per-device summaries

Current aggregate GPU summaries are insufficient for placement optimization.

Add per-device summary metrics:

~~~text
telemetry.gpu.<device>.utilization_avg_pct
telemetry.gpu.<device>.utilization_peak_pct
telemetry.gpu.<device>.vram_used_peak_bytes
telemetry.gpu.<device>.temperature_peak_c
telemetry.gpu.<device>.power_avg_w
telemetry.gpu.<device>.power_peak_w
~~~

Raw samples remain canonical.

### 16.3 Process attribution limitation

GPU-wide telemetry cannot fully attribute utilization to a particular model process on every backend. V2 should report per-device totals without pretending they are per-process unless a provider supplies reliable process attribution.

The V2-M3 implementation uses normalized PCI identity first, UUID second, and explicit stable mappings third for correlation. It never merges by product name alone. Provider provenance is retained on raw GPU samples, aggregate V1 summary fields remain unchanged, and stable encoded per-device metric keys are emitted alongside a structured per-device summary.

## 17. Persistence

Add normalized records rather than embedding the entire deployment result in JSON only.

Recommended tables:

### deployment_candidate

~~~text
id
deployment_hash
definition_json
created_at
~~~

### deployment_instance

~~~text
deployment_candidate_id
instance_id
candidate_id
binary_id
model_artifact_id
role
~~~

### deployment_placement

~~~text
id
deployment_candidate_id
host_id
placement_hash
status
request_json
result_json
created_at
~~~

### deployment_instance_placement

~~~text
deployment_placement_id
instance_id
resolved_placement_id
~~~

### placement_device_memory

~~~text
id
deployment_placement_id
instance_id
device_id
model_bytes
context_bytes
compute_bytes
total_bytes
device_total_bytes
device_free_bytes
source
measured_at
~~~

### deployment_run

~~~text
id
deployment_candidate_id
deployment_placement_id
workload_case_id
status
quality
started_at
finished_at
failure_kind
~~~

### deployment_run_member

~~~text
deployment_run_id
instance_id
server_run_id
client_run_id
endpoint
member_status
pid
argv_json
target_model_path
draft_model_path
started_at
ready_at
finished_at
exit_code
stdout
stderr
forced_kill
cleanup_error
result_json
~~~

V2-M5 migration 010 adds the explicit server lifecycle fields above. They preserve enough evidence to diagnose startup/readiness/cleanup behavior independently for every resident model instance while retaining `result_json` for extensible member metadata.

V2-M6 migrations 011 and 012 add:

- `deployment_concurrent_workload_case` for immutable generated phase definitions;
- `deployment_workload_run` for phase status, quality, barrier/overlap timing, aggregate token counts/TPS, minimum retention, correctness, and failure evidence;
- `deployment_workload_member` for per-instance native/overlap throughput, timing, exact baseline reference, retention/loss, latency delta, correctness, and raw evidence;
- `deployment_standalone_baseline` for exact standalone denominators.

Raw member JSON always includes serialized cumulative token events so the overlap calculation remains auditable even for non-production client implementations.

Existing generic metric and telemetry storage should be reused where practical.

## 18. Placement cache identity

A memory estimate or resolved placement cache key must include:

- exact model artifact identity;
- exact llama.cpp/helper binary SHA-256;
- host hardware fingerprint;
- visible/selected device identities and ordering;
- context size;
- K/V cache type;
- batch and ubatch;
- flash-attention mode;
- split mode;
- GPU layer limit;
- tensor split;
- tensor overrides;
- relevant extra arguments.

Deployment-level feasibility cache identity additionally includes every resident instance because free memory observed during fitting is not a durable substitute for explicit joint allocation.

## 19. CLI

Implemented through V2-M6:

~~~text
llprof deployment preview deployment-plan.json
llprof deployment plan deployment-plan.json
llprof deployment execute deployment-execution.json
llprof deployment benchmark deployment-benchmark.json
~~~

The M5 execution specification identifies one persisted `deployment_placement_id` and supplies the concrete model path for every instance. Optional fields select bind host, readiness timeout, residency hold duration, and per-instance draft model paths. Execution persists the actual endpoints selected by the collision-safe port allocator.

The M6 benchmark specification reuses those placement/model inputs and may add exact standalone baselines. Each baseline names the instance, prefill/decode mode, prompt/generate counts, depth, standalone TPS, and optional latency. `deployment benchmark` keeps the same servers resident, executes the generated DD/PP/PD/DP phase matrix, persists member/aggregate timing and retention evidence, and prints phase quality plus combined PP/TG TPS and minimum retention.

Later milestones add deployment result exploration, resume/cancel surfaces, and Pareto analysis without changing the M5/M6 execution record semantics.

Planning output should show:

- raw candidate count;
- capability-rejected count;
- memory-rejected count;
- valid candidate count;
- estimated benchmark count.

Placement output should show a device-by-instance memory matrix.

Example:

~~~text
                    CUDA0      Vulkan0
Qwen model          10.8 GiB     4.0 GiB
Qwen context         3.1 GiB     1.2 GiB
Qwen compute         0.6 GiB     0.3 GiB
Flash model          0.9 GiB     6.2 GiB
Flash context        0.2 GiB     1.1 GiB
Flash compute        0.1 GiB     0.4 GiB
Reserved             0.8 GiB     0.8 GiB
Projected free       ...
~~~

## 20. API

Expose deployment resources under a separate namespace rather than overloading V1 Candidate endpoints.

Suggested routes:

~~~text
POST /api/deployments
POST /api/deployments/{id}/plan
POST /api/deployments/{id}/run
POST /api/deployments/{id}/resume
POST /api/deployments/{id}/cancel
GET  /api/deployments/{id}
GET  /api/deployments/{id}/candidates
GET  /api/deployments/{id}/placements
GET  /api/deployments/{id}/runs
GET  /api/deployments/{id}/pareto
~~~

SSE progress should include deployment-run state and per-instance state.

## 21. UI

Add a deployment workflow after the V1 single-model workflow remains stable.

Required views:

1. Deployment definition:
   - choose two or more existing base Candidates;
   - assign instance roles;
   - choose allowed devices;
   - define memory margins and constraints.

2. Placement matrix:
   - rows for model/context/compute per instance;
   - columns for devices;
   - remaining headroom;
   - clear infeasibility explanations.

3. Concurrent workload matrix:
   - DD, PP, PD, DP;
   - context depth;
   - per-instance and combined rates.

4. Interference view:
   - standalone versus concurrent throughput;
   - retention percentages;
   - latency deltas.

5. Pareto view:
   - aggregate TPS;
   - minimum retention;
   - context capacity;
   - memory headroom;
   - power.

The UI must never hide the per-model result behind only a combined number.

## 22. Promotion

V2 promotion produces a deployment proposal containing all model profiles and their coordinated placements.

Promotion must refuse to emit a partial deployment when the proposal depends on coordinated memory allocation.

The proposal shall include:

- source launcher snapshot;
- every affected model profile;
- model and binary fingerprints;
- selected devices and placement arguments;
- context and KV settings;
- validated concurrent workload results;
- per-device memory headroom;
- patch/diff for review.

Applying production configuration remains outside automatic benchmark execution.

## 23. Failure model

Add deployment-specific failure kinds:

~~~text
device_capability_mismatch
memory_projection_failed
memory_infeasible
server_start_failed
server_oom
concurrent_workload_failed
member_timeout
member_crash
telemetry_incomplete
runtime_memory_margin_violated
output_validation_failed
~~~

Failures are persisted and remain queryable in analysis.

## 24. Correctness and output validation

Throughput is not sufficient evidence of a usable backend.

Finalist validation should support deterministic or task-specific correctness probes, especially for new backend/device combinations and long-context configurations.

A placement that is fast but produces invalid output is rejected.

Correctness policy is workload-specific and should be implemented as validation hooks rather than a global text-equality rule.

## 25. Compatibility with V1

V2 migrations must be additive.

Existing:

- candidate;
- search space;
- workload suite;
- resolved placement;
- benchmark run;
- server validation;
- analysis;
- promotion;

remain valid.

A single-model experiment does not need to be wrapped in a DeploymentCandidate.

Shared services should be reused where their semantics are genuinely identical. V1 APIs and CLI behavior must not change merely to fit the V2 abstraction.

## 26. Implementation milestones

### V2-M1: domain and persistence

- DeploymentCandidate and ModelInstanceCandidate.
- Additive SQLite migrations.
- Repository round-trip tests.
- Content-addressed identity tests.

### V2-M2: device inventory and memory estimator

- Stable accelerator identity.
- Registered memory-estimation helper.
- common_get_device_memory_data normalization.
- placement_device_memory persistence.
- feasibility checks.

### V2-M3: mixed-vendor telemetry

- Composite NVIDIA plus DRM/sysfs provider.
- device de-duplication.
- per-device summary metrics.
- tests with synthetic NVIDIA and AMD devices in one sample.

### V2-M4: deployment planner

- joint placement dimensions;
- per-device memory margins;
- deterministic pruning;
- rejection reasons;
- planning preview.

### V2-M5: multi-server execution

- simultaneous managed llama-server instances;
- readiness barrier;
- complete cleanup/cancellation;
- stale-run recovery.

### V2-M6: concurrent workloads

- DD, PP, PD, DP orchestration;
- overlap-aware aggregate TPS;
- standalone/concurrent retention metrics;
- failure semantics.

### V2-M7: deployment analysis

- deployment comparison;
- device memory projections;
- interference analysis;
- Pareto frontiers.

### V2-M8: API and CLI

- deployment CRUD/planning/execution/results;
- SSE progress;
- memory matrix rendering.

### V2-M9: browser UI

- deployment editor;
- placement matrix;
- concurrent workload view;
- interference view;
- deployment Pareto view.

### V2-M10: promotion and hardening

- coordinated launcher proposal;
- archive/export support;
- migration tests;
- OOM/corruption/failure-path tests;
- target-workstation acceptance.

## 27. Acceptance criteria

The V2 extension is accepted on the target workstation when all of the following are demonstrated:

1. The lab discovers both GPUs with stable identities.
2. NVIDIA and AMD telemetry appear in the same samples.
3. A Qwen3.8 27B configuration can be evaluated with weights/context split across the allowed devices when supported by the selected llama.cpp build.
4. Flash Next can remain resident simultaneously.
5. Per-instance, per-device model/context/compute memory is persisted.
6. At least one candidate using both GPUs for both resident models is evaluated.
7. DD, PP, PD, and DP workloads produce per-instance and aggregate metrics.
8. Standalone-to-concurrent throughput retention is reported.
9. Memory-infeasible candidates are rejected before full benchmark execution.
10. Runtime OOM or margin violations are persisted rather than lost.
11. Pareto analysis can select candidates using combined decode TPS, combined prompt-processing TPS, minimum retention, context capacity, and VRAM headroom.
12. Finalists pass correctness validation and production-like simultaneous server validation.
13. Existing V1 tests and single-model workflows continue to pass.

## 28. Reference optimization scenario

A useful first reference experiment is:

~~~text
Devices:
  NVIDIA RTX 4070 Ti Super, 16 GiB
  AMD GPU, discovered from the actual llama.cpp build

Instance A:
  Qwen3.8 27B
  weight quantization fixed by selected GGUF
  context sweep: 120K and higher feasible values
  KV sweep: q4_0, q8_0, f16 where supported
  placement: single-device and layer-split candidates

Instance B:
  Flash Next
  production context target
  placement: single-device and layer-split candidates

Objectives:
  maximize DD combined decode TPS
  maximize PP combined prompt-processing TPS
  maximize minimum throughput retention
  maximize validated Qwen context
  maximize minimum device-memory headroom

Constraints:
  both servers resident simultaneously
  no swap
  no unexpected CPU fallback unless explicitly allowed
  clean correctness validation
~~~

The experiment should include the intuitive baseline of one model per GPU. The optimizer succeeds only if it can compare that baseline fairly against coordinated split placements rather than assuming either topology is superior.

## 29. Open questions

The implementation should resolve these through measured capability and experiments rather than assumptions:

- Which backend combination is most reliable for the exact NVIDIA/AMD host?
- Does a single llama.cpp build expose both accelerators in a usable multi-device configuration, or are separate backend-specific binaries preferable?
- At what split ratios does PCIe transfer overhead dominate the benefit of additional VRAM?
- How much compute-buffer memory changes under concurrent server load compared with no-alloc estimates?
- Which model should receive the faster GPU for latency-sensitive phases?
- Is asymmetric placement preferable, for example Qwen primarily on one GPU and Flash primarily on the other with only enough cross-placement to satisfy context-memory goals?
- Which KV precision offers the best context/quality/performance tradeoff for each model?
- When, if ever, should experimental tensor parallelism enter the search space on this heterogeneous platform?

These are benchmark questions and should remain visible as such in the lab.
