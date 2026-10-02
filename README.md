# llama-profile-lab

`llama-profile-lab` is a local experiment, benchmarking, and tuning environment for llama.cpp profiles.

The project is designed around reproducible N-dimensional parameter sweeps, production-context-aware placement, durable SQLite experiment history, CPU/GPU telemetry, and later end-to-end llama-server validation.

## Status

V1 is under active development. The repository currently implements **M0 — Repository bootstrap** and **M1 — Domain model and canonical identities** from the implementation roadmap.

Project documents:

- [V1 technical specification](docs/technical-spec-v1.md)
- [V1 implementation roadmap](docs/implementation-roadmap-v1.md)

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12+ (managed automatically by uv when needed)

llama.cpp is not required for the bootstrap milestone. Later milestones will integrate `llama-bench`, `llama-fit-params`, and `llama-server`.

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

The bootstrap milestone intentionally exposes only the top-level CLI shell. Commands are added as their backing capabilities are implemented.

~~~text
llprof --help
llprof --version
~~~

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
