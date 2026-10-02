# llama-profile-lab 1.0.0 release notes

V1 completes the local profile-tuning lifecycle from launcher configuration through reproducible experiment planning, placement-aware llama-bench execution, CPU/GPU telemetry, multidimensional analysis, llama-server/SPEED-Bench finalist validation, review-first launcher promotion, and archival recovery.

## Highlights

- Immutable Candidate/SearchSpace/WorkloadSuite/MeasurementPolicy models with deterministic content identity.
- SQLite migrations and append-only experiment/run provenance.
- Generic N-dimensional grid expansion with constraints and exact workload expansion.
- Exact executable SHA-256 registration plus help-derived native/custom capability detection.
- Resumable llama-bench execution with persisted repetitions, failures, stdout/stderr, argv, and environment.
- Production-context-aware llama-fit-params placement resolution and reuse.
- CPU/process/RAM/GPU telemetry and run-quality classification.
- Matrix/facet analysis, baseline deltas, Pareto frontiers, latency estimation, and CSV/JSON export.
- Managed llama-server + SPEED-Bench validation including speculative-decoding metrics.
- Local FastAPI/SSE API and React/TypeScript browser workflow.
- Server-validated Candidate promotion proposals with launcher snapshots and unified patches.
- Full experiment provenance export and hashed SQLite archive creation/restoration.
- Database integrity, growth, foreign-key, row-count, and representative query-plan diagnostics.

## M13 hardening

The V1 hardening suite explicitly covers:

- every historical migration prefix upgrading to the current schema;
- interruption/resume and stale running-state recovery;
- persisted OOM failures;
- malformed llama-bench JSON;
- unsupported argument/capability surfaces;
- native versus custom/Qwen-like binary capabilities;
- managed server startup failure;
- noisy and incomplete telemetry;
- launcher source-profile drift after experiment creation;
- archive hash/integrity restore checks;
- database growth measurement and indexed representative query plans.

## Release gate

The source/package version is `1.0.0`. The Git tag `v1.0.0` should be created only after the primary-workstation acceptance checklist passes with the intended production Qwen models, target llama.cpp builds, and workstation hardware.

See:

- `docs/release-checklist-v1.md`
- `docs/benchmark-workflow-v1.md`
- `docs/troubleshooting-v1.md`
- `docs/known-limitations-v1.md`
