# llama-profile-lab

`llama-profile-lab` is a local experiment, benchmarking, and tuning environment for llama.cpp profiles.

The project is designed around reproducible N-dimensional parameter sweeps, production-context-aware placement, durable SQLite experiment history, CPU/GPU telemetry, and later end-to-end llama-server validation.

## Status

V1 implementation is complete through M13. **The automated hardening/release suite is complete; the final primary-workstation acceptance and the `v1.0.0` tag remain intentionally pending until the required Qwen models and target llama.cpp builds are exercised on that workstation.**

Project documents:

- [V1 technical specification](docs/technical-spec-v1.md)
- [V1 implementation roadmap](docs/implementation-roadmap-v1.md)
- [V1 architecture](docs/architecture-v1.md)
- [V1 benchmark workflow](docs/benchmark-workflow-v1.md)
- [V1 troubleshooting](docs/troubleshooting-v1.md)
- [V1 known limitations](docs/known-limitations-v1.md)
- [V1 release checklist](docs/release-checklist-v1.md)

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12+ (managed automatically by uv when needed)
- Node.js 22+ and npm for frontend development/builds

llama.cpp is optional for development and automated tests. M4 discovers exact local binaries, M5 executes planned microbenchmarks, M6 resolves full-production-context placement through a registered `llama-fit-params` binary before `llama-bench` runs, M7 records CPU/process/RAM/GPU telemetry plus run-quality signals, M8 derives statistics and multidimensional comparisons, M9 manages finalist `llama-server` sessions plus SPEED-Bench validation including speculative-decoding acceptance metrics, M10 exposes those same services through a local FastAPI/SSE interface, and M11 provides the browser UI over that service boundary.

## Development setup

Synchronize the project environment from the committed lockfile:

~~~bash
uv sync --frozen
~~~

Run the quality gates through the locked environment:

~~~bash
uv run --frozen ruff check .
uv run --frozen mypy src
uv run --frozen pytest

cd frontend
npm ci
npm run typecheck
npm run test
npm run build
~~~

Verify the CLI entry point:

~~~bash
uv run --frozen llprof --help
uv run --frozen llprof --version
~~~

Dependency policy:

- `pyproject.toml` defines project dependencies and allowed version ranges.
- `uv.lock` is committed and defines the exact Python development/CI resolution.
- `frontend/package-lock.json` is committed and defines the exact frontend development/CI resolution.
- development tools live in the standardized `dev` dependency group.
- setuptools remains the Python build backend; uv manages Python, environments, dependency resolution, and command execution.

## Repository layout

~~~text
src/llama_profile_lab/
  domain/       Domain models and immutable experiment definitions
  planning/     Search-space and workload expansion
  execution/    Scheduling, subprocesses, locking, and telemetry
  llama/        llama.cpp adapters and capability discovery
  db/           SQLite connection, migrations, and repositories
  analysis/     Metrics, projections, Pareto analysis, and latency models
  api/          Local FastAPI application
  cli/          llprof command-line interface

migrations/     Numbered SQLite schema migrations
frontend/       React + TypeScript browser UI
tests/          Unit and integration tests
docs/           Technical specification and roadmap
~~~

## CLI

The planner milestone exposes persisted experiment planning in addition to the top-level CLI shell:

~~~text
llprof --help
llprof --version
llprof experiment plan EXPERIMENT_ID --database data/benchmarks.db

llprof experiment run EXPERIMENT_ID \
  --binary BENCH_BIN_ID \
  --fit-binary FIT_BIN_ID \
  --model-path /path/to/model.gguf \
  --telemetry-interval-ms 1000 \
  --database data/benchmarks.db

llprof experiment resume EXPERIMENT_ID \
  --binary BENCH_BIN_ID \
  --fit-binary FIT_BIN_ID \
  --model-path /path/to/model.gguf \
  --database data/benchmarks.db

llprof placement list --database data/benchmarks.db
llprof placement show PLACEMENT_ID --database data/benchmarks.db
llprof run show RUN_ID --database data/benchmarks.db

llprof results matrix EXPERIMENT_ID \
  --x compute.batch_size \
  --y compute.ubatch_size \
  --metric throughput.median \
  --filter workload.kind=microbench-prefill \
  --filter workload.prompt_tokens=8192 \
  --database data/benchmarks.db

llprof results compare EXPERIMENT_ID CANDIDATE_ID \
  --metric throughput.median \
  --database data/benchmarks.db

llprof results pareto EXPERIMENT_ID \
  --objective 'pp8k:max:throughput.median@workload.kind=microbench-prefill;workload.prompt_tokens=8192' \
  --objective 'tg4k:max:throughput.median@workload.kind=microbench-decode;workload.depth_tokens=4096' \
  --database data/benchmarks.db

llprof results export EXPERIMENT_ID \
  --format csv \
  --output results.csv \
  --database data/benchmarks.db


llprof binary inspect \
  /path/to/llama-server \
  /path/to/speed_bench.py \
  --database data/benchmarks.db

llprof server validate EXPERIMENT_ID CANDIDATE_ID \
  --server-binary SERVER_BIN_ID \
  --speed-bench-binary SPEED_BIN_ID \
  --model-path /path/to/target.gguf \
  --draft-model-path /path/to/draft.gguf \
  --placement PLACEMENT_ID \
  --model-name finalist \
  --database data/benchmarks.db

llprof server compare EXPERIMENT_ID BASELINE_CANDIDATE_ID SPEC_CANDIDATE_ID \
  --category all \
  --database data/benchmarks.db


llprof api \
  --host 127.0.0.1 \
  --port 8000 \
  --launcher-config /path/to/llama-profile-launcher/config/hosts/workstation.json \
  --database data/benchmarks.db

llprof ui \
  --host 127.0.0.1 \
  --port 8000 \
  --launcher-config /path/to/llama-profile-launcher/config/hosts/workstation.json \
  --database data/benchmarks.db

llprof profile promote EXPERIMENT_ID CANDIDATE_ID \
  --launcher-config /path/to/llama-profile-launcher/config/hosts/workstation.json \
  --output candidate.patch \
  --database data/benchmarks.db

llprof experiment export EXPERIMENT_ID \
  --output experiment.json \
  --database data/benchmarks.db

llprof archive \
  --output experiment-archive.tar.gz \
  --artifact /path/to/optional/artifact \
  --database data/benchmarks.db

llprof archive \
  --restore experiment-archive.tar.gz \
  --artifacts-dir restored-artifacts \
  --database data/restored.db

llprof database check \
  --database data/benchmarks.db
~~~

Binary discovery fingerprints exact executables by SHA-256, captures version/help output, persists parsed supported arguments, and allows native/custom builds to be compared without assuming a global llama.cpp feature set.

The plan command expands the stored SearchSpace and WorkloadSuite atomically and does not launch llama.cpp. Microbenchmark workloads create `benchmark_case` rows for M5; server-only `speed-bench` workloads remain persisted in `experiment_workload` and are consumed by M9 rather than being misrouted through llama-bench.

The M5 executor runs incomplete cases sequentially under a host lock, revalidates registered executable SHA-256 identities, persists stdout/stderr/raw JSON and individual repetitions, and resumes only cases without a successful prior run.

M6 adds production-context placement resolution. Under the default per-candidate policy, `llama-fit-params` is run once at the Candidate's full context and its concrete placement is cached and applied to every PP/TG workload for that Candidate. Fixed-placement experiments can deliberately reuse an existing placement on the same host/context. Fit attempts and successful resolved placements are persisted separately.

M7 samples telemetry before, during, and after each benchmark. Linux CPU/process/RAM metrics come from `/proc` and `/sys`; NVIDIA GPUs use `nvidia-smi` when available, with a generic DRM/sysfs fallback. Raw samples are retained, normalized `telemetry.*` summary metrics are generated, and runs receive a quality label such as `clean`, `external_cpu_load`, or `telemetry_incomplete` without changing benchmark success status.

M8 adds a read-only analysis layer over SQLite. Individual repetitions drive mean/median/stddev/CV throughput statistics; resource metrics can be projected over arbitrary Candidate dimensions; higher dimensions use exact filters and facets; ambiguous hidden coordinates are rejected instead of silently averaged. Baseline comparisons report signed deltas, Pareto analysis returns the non-dominated set for caller-defined maximize/minimize objectives, and PP/TG curves can estimate compute-only request latency. CSV and JSON export use the same services as the CLI and future API/UI.

M9 validates selected finalists under a managed `llama-server`. Exact server and SPEED-Bench executables are fingerprinted and capability-checked, the existing resolved placement is frozen into the server argv, readiness is detected through `/health`, server and benchmark subprocess logs/results are persisted independently, and the server process group is cleaned up on completion or failure. SPEED-Bench normalizes prompt/decode throughput, latency, draft/accepted token counts, and acceptance rate while preserving its raw JSON. Candidate evaluation events record the finalist and server-validated stages append-only.

M10 adds a local FastAPI boundary over the same services. Typed HTTP requests can create and plan experiments, inspect/register binaries, run/pause/resume/cancel benchmark execution, inspect Candidates/runs/telemetry, query M8 results and sparse matrices, and invoke M9 Candidate validation. Server-Sent Events expose changed progress snapshots without introducing a second durable scheduler. The server binds to `127.0.0.1` by default. An optional read-only `--launcher-config` exposes current launcher model/profile settings; launcher mutation/promotion remains a later milestone. Interactive API documentation is served at `/api/docs`.

M11 adds the React + TypeScript browser workflow. The Python launcher adapter supplies the typed base Candidate, while the UI builds generic N-dimensional sweeps from backend parameter metadata and asks the real planner for preview counts. It authors microbenchmark and optional SPEED-Bench workloads, controls execution, and visualizes live CPU/GPU/memory/thermal telemetry. Results include sparse matrices/heatmaps with exact higher-dimensional slicing or facets, Pareto frontiers, measured decode-depth curves, baseline-relative Candidate comparison, compute-only latency estimates, and finalist server validation. `llprof ui` serves the production bundle and API from one local origin. The browser never reads SQLite directly and does not silently average hidden search dimensions.

M12 closes the experiment lifecycle without silently editing production configuration. A server-validated Candidate can generate and persist a launcher-profile proposal containing the exact source snapshot, proposed snapshot, argument-level changes, validation provenance, and a unified JSON patch. The Candidate page exposes the same review-first workflow. `llprof experiment export` emits complete experiment provenance, while `llprof archive` checkpoints WAL, takes a consistent SQLite backup, optionally includes artifacts, and writes a SHA-256 manifest.

M13 hardens the V1 boundary. Archive restore now verifies safe member paths, manifest hashes, SQLite integrity, and schema version; promotion rejects source-profile drift after experiment creation; the test suite covers historical migration upgrades, persisted OOM failures, malformed benchmark JSON, unsupported argument surfaces, native/custom capability differences, server startup failure, noisy telemetry, and stale running-state recovery. `llprof database check` reports integrity, foreign keys, database/WAL growth, row counts, and representative SQLite query plans. Release/operator documentation lives under `docs/`.

## Design principles

- SQLite is the canonical experiment record.
- Experiments are planned before execution.
- Candidates and workload cases are immutable.
- Completed runs are append-only.
- Failures are data.
- Candidate search and workload generation are independent.
- llama.cpp CLI details stay behind adapters.
- CPU and GPU are first-class resource measurements.
- CLI and web UI share the same application/domain services.

See the technical specification for the complete architecture.
