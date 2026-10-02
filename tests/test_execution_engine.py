"""End-to-end sequential llama-bench execution and resume tests."""

from __future__ import annotations

from pathlib import Path

from llama_profile_lab.db import (
    BenchmarkCaseRepository,
    BenchmarkRunRepository,
    Database,
    EnvironmentRepository,
    ExperimentRepository,
)
from llama_profile_lab.execution import ExperimentExecutor
from llama_profile_lab.execution.engine import _process_failure_status
from llama_profile_lab.execution.process import ProcessResult
from llama_profile_lab.llama import probe_binary
from llama_profile_lab.planning import plan_experiment
from tests.test_planning_persistence import seed_reference_experiment


def write_fake_llama_bench(path: Path) -> None:
    """Write a fast fake that exposes the M5 option surface and JSON output."""
    script = r'''#!/usr/bin/env python3
import json
import sys

HELP = """usage: llama-bench [options]
  --model PATH
  --n-prompt N
  --n-gen N
  --n-depth N
  --batch-size N
  --ubatch-size N
  --cache-type-k TYPE
  --cache-type-v TYPE
  --threads N
  --flash-attn MODE
  --load-mode MODE
  --lazy-mode MODE
  --repack N
  --repetitions N
  --output FORMAT
  --no-warmup
  --delay N
"""

if "--version" in sys.argv:
    print("version: 9999 (abcdef123)")
    raise SystemExit(0)
if "--help" in sys.argv or "-h" in sys.argv:
    print(HELP)
    raise SystemExit(0)


def option(name, default):
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    return sys.argv[index + 1]


repetitions = int(option("--repetitions", "3"))
n_prompt = int(option("--n-prompt", "0"))
n_gen = int(option("--n-gen", "0"))
n_depth = int(option("--n-depth", "0"))
tokens = n_prompt + n_gen
samples_ns = [1_000_000_000 + index * 1_000_000 for index in range(repetitions)]
samples_ts = [1_000_000_000 * tokens / elapsed for elapsed in samples_ns]
avg_ns = sum(samples_ns) // len(samples_ns)
avg_ts = sum(samples_ts) / len(samples_ts)

print(json.dumps([{
    "build_commit": "abcdef123",
    "build_number": 9999,
    "n_prompt": n_prompt,
    "n_gen": n_gen,
    "n_depth": n_depth,
    "avg_ns": avg_ns,
    "stddev_ns": 0,
    "avg_ts": avg_ts,
    "stddev_ts": 0.0,
    "samples_ns": samples_ns,
    "samples_ts": samples_ts,
}]))
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def register_fake_bench(database: Database, binary: Path) -> str:
    probe = probe_binary(binary)
    with database.session() as connection:
        return EnvironmentRepository(connection).put_binary(
            sha256=probe.sha256,
            kind=probe.kind,
            path=str(probe.path),
            size_bytes=probe.size_bytes,
            mtime_ns=probe.mtime_ns,
            git_commit=probe.git_commit,
            build_number=probe.build_number,
            build_info=probe.build_info_mapping(),
            capabilities=probe.capabilities.to_mapping(),
        )


def test_reference_experiment_runs_five_then_resumes_remaining_39(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "benchmarks.db")
    experiment_id = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, experiment_id)

    binary = tmp_path / "llama-bench"
    write_fake_llama_bench(binary)
    binary_id = register_fake_bench(database, binary)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake-model")

    executor = ExperimentExecutor(database)
    first = executor.execute(
        experiment_id,
        binary_id=binary_id,
        model_path=model,
        limit=5,
    )

    assert first.attempted == 5
    assert first.completed == 5
    assert first.failed == 0
    assert first.remaining == 39
    assert first.limited

    with database.session() as connection:
        record = ExperimentRepository(connection).get(experiment_id)
        assert record is not None
        assert record.status == "paused"

    second = executor.execute(
        experiment_id,
        binary_id=binary_id,
        model_path=model,
        resume=True,
    )

    assert second.attempted == 39
    assert second.completed == 39
    assert second.failed == 0
    assert second.remaining == 0
    assert not second.limited

    with database.session() as connection:
        record = ExperimentRepository(connection).get(experiment_id)
        assert record is not None
        assert record.status == "completed"

        run_count = connection.execute(
            "SELECT COUNT(*) FROM benchmark_run WHERE status = 'completed'"
        ).fetchone()[0]
        sample_count = connection.execute(
            "SELECT COUNT(*) FROM benchmark_sample"
        ).fetchone()[0]
        max_successes_per_case = connection.execute(
            """
            SELECT MAX(success_count)
            FROM (
                SELECT benchmark_case_id, COUNT(*) AS success_count
                FROM benchmark_run
                WHERE status = 'completed'
                GROUP BY benchmark_case_id
            )
            """
        ).fetchone()[0]

        assert run_count == 44
        assert sample_count == 132
        assert max_successes_per_case == 1


def test_orphaned_running_attempt_is_recovered_before_resume(tmp_path: Path) -> None:
    database = Database(tmp_path / "orphan.db")
    experiment_id = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, experiment_id)
        cases = BenchmarkCaseRepository(connection)
        first_case = cases.list_incomplete(experiment_id)[0]

        environment = EnvironmentRepository(connection)
        host_id = environment.put_host(
            hostname="host",
            hardware_fingerprint="test-hardware",
            cpu={},
            ram_bytes=0,
            gpus=[],
            os_info={},
        )
        binary_id = environment.put_binary(
            sha256="1" * 64,
            kind="llama-bench",
            path="/fake/llama-bench",
            size_bytes=1,
            mtime_ns=1,
        )
        policy_id = ExperimentRepository(connection).get(experiment_id)
        assert policy_id is not None

        definition = ExperimentRepository(connection).get_definition(experiment_id)
        assert definition is not None
        runs = BenchmarkRunRepository(connection)
        run_id = runs.create(
            benchmark_case_id=first_case.id,
            host_id=host_id,
            binary_id=binary_id,
            measurement_policy_id=definition.measurement_policy_id,
            argv=("llama-bench",),
            environment={},
        )
        cases.set_status(first_case.id, "running")

        recovered = runs.recover_orphaned(experiment_id)

        assert recovered == 1
        run = runs.get(run_id)
        assert run is not None
        assert run.status == "interrupted"
        case = cases.get(first_case.id)
        assert case is not None
        assert case.status == "planned"


def test_process_failure_classification_detects_oom() -> None:
    result = ProcessResult(
        argv=("llama-bench",),
        pid=123,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        duration_ns=1_000_000_000,
        exit_code=1,
        stdout="",
        stderr="CUDA error: out of memory",
    )

    assert _process_failure_status(result) == "oom"
