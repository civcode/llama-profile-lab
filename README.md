# llama-profile-lab

`llama-profile-lab` is a local experiment, benchmarking, and tuning environment for llama.cpp profiles.

The project is designed around reproducible N-dimensional parameter sweeps, production-context-aware placement, durable SQLite experiment history, CPU/GPU telemetry, and later end-to-end llama-server validation.

## Status

V1 is under active development. **M0–M4 are complete. M5 — llama-bench execution engine and M6 — placement resolution are implementation-complete with automated acceptance; their final real-workstation acceptance remains pending.**

Project documents:

- [V1 technical specification](docs/technical-spec-v1.md)
- [V1 implementation roadmap](docs/implementation-roadmap-v1.md)

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12+ (managed automatically by uv when needed)

llama.cpp is optional for development and automated tests. M4 discovers exact local binaries, M5 executes planned microbenchmarks, and M6 resolves full-production-context placement through a registered `llama-fit-params` binary before `llama-bench` runs.

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
  --database data/benchmarks.db

llprof experiment resume EXPERIMENT_ID \
  --binary BENCH_BIN_ID \
  --fit-binary FIT_BIN_ID \
  --model-path /path/to/model.gguf \
  --database data/benchmarks.db

llprof placement list --database data/benchmarks.db
llprof placement show PLACEMENT_ID --database data/benchmarks.db
llprof run show RUN_ID --database data/benchmarks.db
~~~

Binary discovery fingerprints exact executables by SHA-256, captures version/help output, persists parsed supported arguments, and allows native/custom builds to be compared without assuming a global llama.cpp feature set.

The plan command expands the stored SearchSpace and WorkloadSuite, persists Candidates and concrete benchmark cases atomically, and does not launch llama.cpp.

The M5 executor runs incomplete cases sequentially under a host lock, revalidates registered executable SHA-256 identities, persists stdout/stderr/raw JSON and individual repetitions, and resumes only cases without a successful prior run.

M6 adds production-context placement resolution. Under the default per-candidate policy, `llama-fit-params` is run once at the Candidate's full context and its concrete placement is cached and applied to every PP/TG workload for that Candidate. Fixed-placement experiments can deliberately reuse an existing placement on the same host/context. Fit attempts and successful resolved placements are persisted separately.

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
