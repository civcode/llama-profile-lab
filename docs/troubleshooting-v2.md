# V2 deployment troubleshooting

## Device capability mismatch before startup

Execution re-discovers the exact registered binaries' accelerator inventory before launching servers. This failure means a planned logical selector no longer resolves to the planned physical GPU.

Check for:

- changed driver/backend;
- changed llama.cpp build;
- GPU removal/reordering;
- stale logical-to-physical mapping;
- a registered binary that no longer exposes the expected device.

Re-inspect the exact binaries/devices and create a new plan. Do not force execution with a stale placement.

## Memory estimator rejected or failed

A feasibility rejection is different from helper execution failure.

For estimator failures inspect the persisted attempt stdout/stderr and status. Malformed JSON, nonzero exit, timeout, cancellation, or helper SHA drift are retained separately. Re-register a helper whose file changed.

For genuine memory infeasibility inspect per-device model/context/compute bytes and the configured margin.

## Runtime memory is worse than projected

The memory matrix reports:

- peak-use delta = observed peak used minus projected allocation;
- free-headroom delta = observed minimum free minus projected free.

Positive peak-use error and negative free-headroom error are warning signs.

Ensure the workstation is otherwise idle before attributing the delta to llama.cpp. External GPU consumers affect total device memory telemetry. If repeated clean runs show a systematic estimator bias, document and apply an explicit overhead policy.

## Missing planned-device telemetry

Runtime margin validation fails closed when a planned physical GPU is absent from the telemetry snapshot.

Check provider support and stable physical keys. In heterogeneous systems, verify NVIDIA and DRM/sysfs providers are both contributing and that correlated devices are not accidentally duplicated.

## One server starts and another fails

The deployment is still failed. The executor tears down the full process group set.

Inspect every deployment member's argv, stdout, stderr, readiness state, exit code, forced-kill flag, and cleanup error. OOM-like stderr is normalized to `server_oom`.

## Concurrent phase is correctness-invalid

Throughput is not accepted when output evidence is corrupt.

The production client marks invalid:

- malformed/non-object SSE events;
- regressing cumulative token counters;
- contradictory final stream/timing counters;
- exact token-count mismatches.

The phase is persisted as `output_validation_failed` and excluded from valid Pareto evidence.

## Retention is missing

Retention requires one exact standalone baseline. No match becomes `baseline_missing`; multiple exact matches become `baseline_ambiguous`.

Collect a baseline for the exact Candidate, resolved placement, host, binary, mode, prompt/generate count, and depth. Do not substitute a nearby baseline.

## Pause/resume behaves differently across processes

Deployment operations are persisted in SQLite. A pause/cancel request can come from another CLI/API process. Resume reuses the last persisted request unless you supply a replacement spec.

If state appears stale after a crash, inspect operation and deployment-run rows. Stale running members are recovered by the repository/service recovery path rather than silently treated as completed.

## Coordinated promotion is refused

Common causes:

- a deployment instance has no source experiment mapping;
- the source launcher profile changed since the source experiment;
- the placement has no completed joint run;
- one configured DD/PP/PD/DP phase is missing or correctness-invalid;
- memory-estimator/helper provenance is missing.

Create new source experiments after launcher drift instead of applying a stale proposal. Promotion is review-only and never edits the launcher file.

## Archive restore fails

Restore verifies:

- safe member paths;
- manifest file set;
- file size and SHA-256;
- SQLite integrity;
- schema version.

Do not modify an archive in place. Produce a new archive from the source database if any verification fails.

## Need a support artifact

Export complete V2 provenance:

```bash
llprof deployment export DEPLOYMENT_ID \
  --output deployment-provenance.json \
  --database data/benchmarks.db
```

This is the preferred single-workflow diagnostic artifact in addition to a full database archive.
