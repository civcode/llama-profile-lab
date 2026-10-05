# V2 multi-model optimizer release notes — draft

Status: **implementation candidate; automated clean-checkout gates and target-workstation acceptance are still pending.**

These notes describe the implemented V2 surface. They are not a release declaration and intentionally contain no unmeasured workstation performance claims.

## Joint multi-model planning

V2 introduces immutable DeploymentCandidates that combine two or more V1 Candidates with exact model artifacts, server binaries, resource policy, and concurrent workload phases.

The joint planner supports instance-addressable context/KV/compute and placement dimensions, deterministic grid expansion, constraints, stable logical-to-physical GPU identity, structured per-instance memory estimation, aggregate per-device feasibility, explicit memory margins, preview counts, persisted feasible cases, and explainable rejection history.

## Heterogeneous accelerator identity and telemetry

Exact binary device inventories are correlated to stable physical identities without merging devices by product name alone. Mixed provider telemetry can retain NVIDIA and DRM/sysfs observations together.

Execution now re-discovers exact-binary device inventory immediately before server startup. A disappeared logical selector or physical-device remapping fails closed as `device_capability_mismatch`.

## Simultaneous residency and DD/PP/PD/DP

One deployment execution owns all resident llama-server process groups. Every member must become ready before synchronized concurrent work begins, and terminal failure tears down the full set.

Concurrent workloads use a shared barrier and retain raw token-progress timing. Aggregate throughput is overlap-normalized rather than produced by simply summing independent measurements.

Exact standalone baselines provide per-instance retention and latency deltas. Missing or ambiguous denominators remain explicit quality states.

## Correctness before throughput

Concurrent output validation rejects malformed/non-object SSE events, regressing cumulative token counters, contradictory final counters, and exact token-count mismatches.

Correctness-invalid rows remain exportable but are excluded from valid Pareto evidence.

## Memory analysis

The deployment memory matrix separates projected model/context/compute/margin values from observed runtime evidence.

V2-M10 adds per-device signed projection deltas:

- observed peak used minus projected allocation;
- observed minimum free minus projected free.

These are diagnostic evidence; the planner does not silently inflate margins.

## Deployment analysis and Pareto

Deployment placements are optimization points backed by repeated run evidence. Exact filters/facets prevent hidden dimensions from being silently averaged.

Pareto objectives can combine DD decode, PP throughput, minimum retention, validated context, runtime headroom, and power with pre-dominance constraints.

## CLI, HTTP/SSE, and browser workflow

Deployment creation, preview/planning, execution controls, inspection, raw results, Pareto analysis, and progress are exposed through shared service boundaries.

Durable schema-14 operations allow pause/cancel/resume across separate processes.

The browser adds:

- deployment authoring from persisted V1 Candidates;
- exact device/resource-policy selection;
- planning preview and pruning explanations;
- projected/runtime memory matrices and deltas;
- live execution state;
- DD/PP/PD/DP tables;
- per-instance interference evidence;
- Pareto filters/scatter;
- Candidate drill-down.

## Coordinated promotion

Schema 15 adds append-only coordinated deployment promotion proposals.

A proposal covers every affected launcher profile together and records:

- source/proposed launcher snapshots;
- source experiment/Candidate provenance;
- exact model artifacts;
- server/helper SHA-256 identities;
- selected/resolved placement arguments;
- projected and runtime memory;
- concurrent correctness/throughput evidence;
- one unified review patch.

Partial proposals, launcher source drift, missing estimator provenance, missing completed phases, or correctness-invalid validation fail closed.

Proposal generation does not mutate launcher configuration.

## Provenance and recovery

`llprof deployment export DEPLOYMENT_ID` emits complete deterministic V2 workflow provenance, including immutable definitions, plan/rejection history, estimator/device evidence, placements, raw run/member/workload evidence, telemetry, durable operations, environment identities, and coordinated promotion proposals.

Schema-15 archives restore V2 tables and promotion evidence after manifest/hash/SQLite validation.

`llprof database check` includes representative indexed deployment plan, run, and promotion queries.

## Release blockers

Before these notes can become final release notes:

- run the locked Python quality gates from a clean checkout;
- run frontend install/typecheck/test/build from a clean checkout;
- execute the target NVIDIA + AMD workstation acceptance plan;
- record hardware, driver, model, binary, and artifact identities;
- measure projected/runtime memory error on the target workloads;
- record one-model-per-GPU and split-placement baselines;
- repeat Pareto finalists;
- verify no hidden swap or unintended CPU fallback;
- update V2 known limitations with measured behavior;
- attach the complete deployment export and database archive for the accepted commit.

See `docs/release-checklist-v2.md`.
