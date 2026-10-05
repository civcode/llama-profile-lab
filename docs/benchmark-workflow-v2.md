# V2 deployment benchmark workflow

This workflow is the operator sequence for repeatable multi-model GPU optimization.

## 1. Freeze inputs

Record:

- source V1 experiment IDs and Candidate IDs;
- model artifact identities and paths;
- exact llama-server and llama-memory-estimator binary IDs/SHA-256 values;
- physical GPU identities, drivers, and backends;
- workload suite and canonical phases;
- CPU-offload, swap, memory-margin, and power policies.

Re-run binary/device inspection after changing a build, driver, GPU topology, or backend.

## 2. Establish standalone evidence

Retention is only meaningful with an exact standalone denominator. Collect baselines for the same:

- Candidate;
- resolved placement;
- host;
- binary;
- prefill/decode mode;
- prompt/generate/depth coordinates.

Missing or ambiguous baselines remain explicit quality states and are not guessed.

## 3. Preview joint feasibility

Build a bounded search over context/KV/placement/device coordinates.

Preview before persistence:

```bash
llprof deployment preview deployment-plan.json \
  --database data/benchmarks.db
```

Review:

- raw combinations;
- constraint rejections;
- duplicate/symmetry reductions;
- capability rejections;
- estimator failures;
- memory rejections;
- feasible count.

A rejected point is useful evidence. Do not widen margins merely to remove a rejection without understanding the runtime overhead.

## 4. Persist the plan

```bash
llprof deployment plan deployment-plan.json \
  --database data/benchmarks.db
```

The persisted plan freezes feasible placement IDs and rejection history.

## 5. Prove simultaneous residency

Use the browser deployment run control or a deployment execution spec. Execution performs binary and device-inventory revalidation before startup.

All members must become ready. After readiness, runtime GPU headroom is checked against the planned reserved margin.

Treat these as hard failures:

- device mapping drift;
- server startup/OOM;
- readiness timeout;
- missing planned-device telemetry;
- runtime memory-margin violation;
- cleanup failure.

## 6. Run synchronized phases

Canonical two-instance phases are:

- DD: both decode;
- PP: both prefill;
- PD: first prefill, second decode;
- DP: first decode, second prefill.

The client records raw token-progress timing and computes overlap-normalized aggregate throughput. A successful HTTP response alone is not sufficient: output/token correctness must also pass.

## 7. Inspect memory accuracy

For each placement compare:

- projected model/context/compute bytes;
- reserved margin;
- projected free bytes;
- runtime peak used;
- runtime minimum free;
- peak-use delta;
- free-headroom delta.

Systematic deltas should become an explicit documented overhead policy. Do not silently bake unexplained padding into search inputs.

## 8. Inspect interference

For every instance and phase record:

- standalone TPS;
- overlap/concurrent TPS;
- retention;
- throughput loss;
- latency;
- latency delta.

Keep deployment aggregate PP/TG beside, not instead of, per-instance evidence.

## 9. Select finalists

Use constrained Pareto analysis. Typical objectives include:

- maximize DD decode TPS;
- maximize PP throughput;
- maximize minimum retention;
- maximize validated context;
- maximize runtime headroom;
- minimize total GPU power.

Use exact filters for phase, context/KV, backend/device placement, and correctness when comparing unlike points.

## 10. Repeat finalists

For workstation acceptance, repeat non-dominated finalists at least three times per canonical phase after the chosen thermal/warmup policy. Require clean telemetry and output correctness.

## 11. Generate coordinated promotion

Use the finalist Candidate page to select source experiment provenance for every instance and generate one coordinated proposal.

Review the unified patch and evidence. Proposal generation never changes launcher configuration.

## 12. Generate the machine-checkable acceptance report

After the baseline, split topology, canonical phases, repetitions, telemetry, and coordinated promotion are persisted, evaluate the evidence already in SQLite:

```bash
llprof deployment acceptance-report DEPLOYMENT_ID \
  --minimum-devices 2 \
  --minimum-phase-repetitions 3 \
  --format json \
  --output workstation-acceptance.json \
  --database data/benchmarks.db
```

The command is read-only. It checks stable mapped device identity, heterogeneous inventory, simultaneous multi-device telemetry, exact server/helper/model provenance, projected/runtime memory evidence, memory-overcommit pruning, correctness-valid DD/PP/PD/DP repetitions, retention, completed baseline and split topologies, no-swap/no-CPU-offload policy, and coordinated promotion coverage.

The report deliberately leaves three items `pending`: explicit Pareto finalist selection, confirmation that the recorded devices/models are the intended target workstation/artifacts, and the clean-checkout backend/frontend gates. It therefore cannot by itself declare a release ready. A nonzero exit status means one or more machine-checkable persisted-evidence requirements failed.

## 13. Export acceptance evidence

Create both:

```bash
llprof deployment export DEPLOYMENT_ID \
  --output deployment-provenance.json \
  --database data/benchmarks.db

llprof archive \
  --output workstation-acceptance.tar.gz \
  --database data/benchmarks.db
```

Keep the acceptance report, deployment export, archive, exact binary hashes, hardware inventory, Pareto selection notes, and operator notes together as release acceptance artifacts.
