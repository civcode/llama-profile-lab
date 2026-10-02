# llama-profile-lab

`llama-profile-lab` is a local experiment, benchmarking, and tuning environment for llama.cpp profiles.

The project is designed around reproducible N-dimensional parameter sweeps, production-context-aware placement, durable SQLite experiment history, CPU/GPU telemetry, and later end-to-end llama-server validation.

## Status

V1 is under active development. The repository currently implements **M0 — Repository bootstrap** from the implementation roadmap.

Project documents:

- [V1 technical specification](docs/technical-spec-v1.md)
- [V1 implementation roadmap](docs/implementation-roadmap-v1.md)

## Requirements

- Python 3.12+

llama.cpp is not required for the bootstrap milestone. Later milestones will integrate `llama-bench`, `llama-fit-params`, and `llama-server`.

## Development setup

Create and activate a virtual environment, then install the package with development dependencies:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

Run the quality gates:

~~~bash
ruff check .
mypy src
pytest
~~~

Verify the CLI entry point:

~~~bash
llprof --help
llprof --version
~~~

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
