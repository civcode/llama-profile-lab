# llama-profile-lab

`llama-profile-lab` is a local experiment, benchmarking, and tuning environment for llama.cpp profiles.

The project is designed around reproducible N-dimensional parameter sweeps, production-context-aware placement, durable SQLite experiment history, CPU/GPU telemetry, and later end-to-end llama-server validation.

## Status

V1 is under active development. The repository currently implements **M0 — Repository bootstrap**, **M1 — Domain model and canonical identities**, **M2 — SQLite persistence and migrations**, and **M3 — Planning engine and N-dimensional expansion** from the implementation roadmap.

Project documents:

- [V1 technical specification](docs/technical-spec-v1.md)
- [V1 implementation roadmap](docs/implementation-roadmap-v1.md)

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12+ (managed automatically by uv when needed)

llama.cpp is optional for development and CI tests. M4 can discover and fingerprint local `llama-bench`, `llama-fit-params`, and `llama-server` executables, and M5 can execute planned microbenchmarks with a registered `llama-bench` binary.

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
  --binary BIN_ID \
  --model-path /path/to/model.gguf \
  --database data/benchmarks.db

llprof experiment resume EXPERIMENT_ID \
  --binary BIN_ID \
  --model-path /path/to/model.gguf \
  --database data/benchmarks.db

llprof run show RUN_ID --database data/benchmarks.db
~~~

Binary discovery fingerprints exact executables by SHA-256, captures version/help output, persists parsed supported arguments, and allows native/custom builds to be compared without assuming a global llama.cpp feature set.

The plan command expands the stored SearchSpace and WorkloadSuite, persists Candidates and concrete benchmark cases atomically, and does not launch llama.cpp.

The M5 executor runs incomplete cases sequentially under a host lock, revalidates the registered binary SHA-256 before execution, persists stdout/stderr/raw JSON and individual repetitions, and resumes only cases without a successful prior run. Placement fitting is intentionally deferred to M6.

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
