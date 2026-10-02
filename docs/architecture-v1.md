# V1 architecture

llama-profile-lab is a local, SQLite-backed experiment system for reproducible llama.cpp tuning. The design keeps configuration, execution, measurement, analysis, and deployment validation separate so historical observations remain explainable.

## Layers

`domain/` defines immutable Candidates, search spaces, workloads, measurement policies, experiment definitions, placements, and telemetry records. Content-addressed domain objects are reused rather than rewritten.

`planning/` expands an N-dimensional SearchSpace, evaluates constraints, expands workload templates at the Candidate production context, and persists the complete Candidate × workload plan before execution.

`llama/` owns executable discovery, SHA-256 identity, help-surface capability detection, and argv translation for llama-bench, llama-fit-params, llama-server, and SPEED-Bench. The scheduler does not contain llama.cpp flag spelling.

`execution/` owns the host lock, subprocess lifecycle, resumability, per-Candidate placement resolution, telemetry sampling, benchmark parsing, and managed server validation.

`db/` owns SQLite configuration, numbered migrations, and repositories. SQLite is the scientific record: immutable definitions, argv, environment snapshots, raw tool output, repetitions, telemetry, failures, validation, and promotion events remain queryable.

`analysis/` derives statistics from persisted observations. Matrix projections require exact hidden-dimension filters or facets; Pareto analysis uses caller-defined objectives and never invents one universal score.

`api/` exposes the same services through typed FastAPI DTOs and SSE. `frontend/` is a React/TypeScript client that uses only HTTP/SSE and never reads SQLite directly.

`promotion.py` generates review-first launcher patches for server-validated Candidates. Promotion is append-only and refuses to proceed if the current launcher performance profile has drifted from the experiment's frozen base Candidate.

`archive.py` creates WAL-checkpointed SQLite backup archives, hashes every included file, and verifies hashes plus SQLite integrity when restoring.

`diagnostics.py` reports schema/integrity state, foreign-key violations, database/WAL size, table row counts, and representative SQLite query plans.

## Core invariants

1. Candidate and workload identity is immutable.
2. Experiment plans freeze before benchmark execution.
3. Completed and failed attempts are append-only observations.
4. Requested placement and resolved host placement are distinct.
5. Production context controls fitting; workload depth controls measurement.
6. Binary capabilities are detected per exact executable hash.
7. Raw observations survive analysis-policy changes.
8. SQLite is the canonical durable record.
9. CLI, HTTP API, and browser UI share the same domain/application services.
10. Promotion proposes changes; it does not silently mutate production launcher configuration.

## Recovery model

A successful benchmark case is not rerun during normal resume. Stale running attempts are converted to `interrupted`, their cases become retryable, and completed work is preserved. OOM, timeout, parse, fit, load, and benchmark failures remain queryable rather than being erased.

Archives are an independent recovery boundary. `llprof archive` checkpoints WAL and uses SQLite's online backup API. `llprof archive --restore` validates the manifest, SHA-256 hashes, member paths, SQLite integrity, and schema version before writing a restored database.
