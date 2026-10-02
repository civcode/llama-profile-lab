# llama-profile-lab

`llama-profile-lab` is a local experiment, benchmarking, and tuning environment for llama.cpp profiles.

The project is designed around reproducible N-dimensional parameter sweeps, production-context-aware placement, durable SQLite experiment history, CPU/GPU telemetry, and later end-to-end llama-server validation.

## Status

V1 is under active development. **M0–M4 and M8 — analysis and multidimensional projections are complete. M5 — llama-bench execution, M6 — placement resolution, M7 — CPU/GPU telemetry/run quality, and M9 — llama-server/SPEED-Bench finalist validation are implementation-complete with automated acceptance; their final real-workstation acceptance remains pending where hardware/model execution is required.**

Project documents:

- [V1 technical specification](docs/technical-spec-v1.md)
- [V1 implementation roadmap](docs/implementation-roadmap-v1.md)

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12+ (managed automatically by uv when needed)

llama.cpp is optional for development and automated tests. M4 discovers exact local binaries, M5 executes planned microbenchmarks, M6 resolves full-production-context placement through a registered `llama-fit-params` binary before `llama-bench` runs, M7 records CPU/process/RAM/GPU telemetry plus run-quality signals, M8 derives statistics and multidimensional comparisons, and M9 manages finalist `llama-server` sessions plus SPEED-Bench validation including speculative-decoding acceptance metrics.

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
~~~

Verify the CLI entry point:

~~~bash
uv run --frozen llprof --help
uv run --frozen llprof --version
~~~

Dependency policy:

- `pyproject.toml` defines project dependencies and allowed version ranges.
- `uv.lock` is committed and defines the exact development/CI resolution.
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
frontend/       React frontend, introduced in a later milestone
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
~~~

Binary discovery fingerprints exact executables by SHA-256, captures version/help output, persists parsed supported arguments, and allows native/custom builds to be compared without assuming a global llama.cpp feature set.

The plan command expands the stored SearchSpace and WorkloadSuite atomically and does not launch llama.cpp. Microbenchmark workloads create `benchmark_case` rows for M5; server-only `speed-bench` workloads remain persisted in `experiment_workload` and are consumed by M9 rather than being misrouted through llama-bench.

The M5 executor runs incomplete cases sequentially under a host lock, revalidates registered executable SHA-256 identities, persists stdout/stderr/raw JSON and individual repetitions, and resumes only cases without a successful prior run.

M6 adds production-context placement resolution. Under the default per-candidate policy, `llama-fit-params` is run once at the Candidate's full context and its concrete placement is cached and applied to every PP/TG workload for that Candidate. Fixed-placement experiments can deliberately reuse an existing placement on the same host/context. Fit attempts and successful resolved placements are persisted separately.

M7 samples telemetry before, during, and after each benchmark. Linux CPU/process/RAM metrics come from `/proc` and `/sys`; NVIDIA GPUs use `nvidia-smi` when available, with a generic DRM/sysfs fallback. Raw samples are retained, normalized `telemetry.*` summary metrics are generated, and runs receive a quality label such as `clean`, `external_cpu_load`, or `telemetry_incomplete` without changing benchmark success status.

M8 adds a read-only analysis layer over SQLite. Individual repetitions drive mean/median/stddev/CV throughput statistics; resource metrics can be projected over arbitrary Candidate dimensions; higher dimensions use exact filters and facets; ambiguous hidden coordinates are rejected instead of silently averaged. Baseline comparisons report signed deltas, Pareto analysis returns the non-dominated set for caller-defined maximize/minimize objectives, and PP/TG curves can estimate compute-only request latency. CSV and JSON export use the same services as the CLI and future API/UI.

M9 validates selected finalists under a managed `llama-server`. Exact server and SPEED-Bench executables are fingerprinted and capability-checked, the existing resolved placement is frozen into the server argv, readiness is detected through `/health`, server and benchmark subprocess logs/results are persisted independently, and the server process group is cleaned up on completion or failure. SPEED-Bench normalizes prompt/decode throughput, latency, draft/accepted token counts, and acceptance rate while preserving its raw JSON. Candidate evaluation events record the finalist and server-validated stages append-only.

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
