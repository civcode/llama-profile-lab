# V2 multi-model deployment architecture

V2 is additive to the V1 local-first experiment architecture. SQLite remains authoritative; the browser, CLI, and HTTP API share the same planning, execution, analysis, and promotion services.

## Resource hierarchy

```text
V1 Candidate
  └─ per-model configuration

DeploymentCandidate
  ├─ instance -> V1 Candidate
  ├─ model artifact
  ├─ exact llama-server binary
  ├─ requested placement
  ├─ resource policy
  └─ workload mix

Deployment plan
  └─ feasible DeploymentPlacement[]
       ├─ per-instance ResolvedPlacement
       ├─ per-instance/per-device memory projection
       └─ per-device aggregate allocation + reserved margin

Deployment run
  ├─ resident server members
  ├─ GPU residency telemetry
  └─ synchronized DD/PP/PD/DP workload runs
       └─ per-instance raw token/timing evidence
```

## Planning boundary

Deployment search expansion is deterministic and content-addressed. Instance-addressable dimensions modify a copy of the base deployment and its referenced V1 Candidates. Constraints are evaluated before expensive estimator work.

For each expanded point the M4 planner:

1. validates Candidate and requested-placement semantics;
2. checks the exact registered server binary capability surface;
3. resolves logical accelerator selectors to stable physical-device identities;
4. runs the exact registered memory-estimator helper per instance;
5. aggregates model/context/compute bytes by physical device;
6. applies device and host memory margins;
7. persists only feasible placements while keeping normalized rejection evidence.

Planning never starts throughput workloads.

## Device identity

Logical names such as `CUDA0` and `Vulkan0` are not treated as durable physical identity. Inventory correlation prefers stable PCI/UUID identity and may use explicit logical-to-physical mappings when a backend cannot expose one.

Immediately before execution, the exact binaries' inventories are discovered again. Every resolved logical selector must still map to the physical allocation persisted by planning. Remapping or disappearance fails before process startup as `device_capability_mismatch`.

## Residency and concurrent execution

One deployment executor owns all server process groups under the host execution lock. The complete set must pass readiness before workload execution starts.

The concurrent executor reuses the resident server lifecycle. A shared barrier releases all clients for a phase. Raw prompt/decode progress events are persisted and overlap throughput is computed only inside the interval where the relevant members overlap.

Output correctness is independent from throughput. Invalid SSE payloads, regressing token counters, contradictory final counters, and exact-token mismatches are correctness failures rather than valid performance observations.

## Memory truth

Planning stores projected model/context/compute memory and per-device aggregate allocations. Residency telemetry stores observed device state.

The analysis memory matrix reports those sources separately and computes signed per-device deltas:

```text
used_delta = runtime_peak_used - projected_allocation
free_delta = runtime_min_free - projected_free
```

These values are evidence, not an automatic margin adjustment. External GPU consumers can affect runtime totals, so acceptance requires a clean workstation.

## Durable operations

Schema 14 stores deployment execution requests and control state. Pause/cancel requests are cooperative and survive process boundaries. Resume reconstructs the persisted request unless an explicit replacement request is supplied.

## Coordinated promotion

Schema 15 stores append-only deployment promotion proposals. Promotion is deliberately separated from execution and never mutates launcher configuration.

A proposal references one finalist deployment placement and requires source experiment/profile provenance for every instance. It validates source-profile drift and correctness-valid concurrent evidence, then creates one proposed launcher snapshot and unified patch spanning all profiles.

## Analysis and Pareto

Deployment placements are the optimization points. Repeated deployment/workload runs are evidence for those points.

Analysis preserves exact hidden coordinates; it does not silently average incompatible phase, depth, backend, or placement dimensions. Failed/correctness-invalid evidence remains exportable but is excluded from valid frontier calculations.

## Provenance and recovery

`llprof deployment export` emits deterministic V2 workflow provenance, including immutable definitions, plans/rejections, estimator/device evidence, placements, runs/members/phases, GPU telemetry, operations, and promotion proposals.

`llprof archive` takes a consistent SQLite backup. Restore validates archive paths, file hashes, SQLite integrity, and schema version before copying the database.
