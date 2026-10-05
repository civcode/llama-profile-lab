"""End-to-end placement-aware llama-bench execution and resume tests."""

from __future__ import annotations

from pathlib import Path

from llama_profile_lab.db import (
    BenchmarkCaseRepository,
    BenchmarkRunRepository,
    Database,
    EnvironmentRepository,
    ExperimentRepository,
)
from llama_profile_lab.domain import ExperimentDefinition, FixedPlacementPolicy
from llama_profile_lab.domain.telemetry import (
    GpuTelemetrySample,
    TelemetryPhase,
    TelemetrySample,
)
from llama_profile_lab.execution import ExperimentExecutor
from llama_profile_lab.execution.engine import _process_failure_status
from llama_profile_lab.execution.process import ProcessResult
from llama_profile_lab.llama import probe_binary
from llama_profile_lab.planning import plan_experiment
from tests.test_planning_persistence import seed_reference_experiment


def write_fake_llama_bench(path: Path) -> None:
    """Write a fast fake that exposes the placement-aware bench option surface."""
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
  --n-gpu-layers N
  --n-cpu-moe N
  --split-mode MODE
  --main-gpu N
  --device LIST
  --tensor-split LIST
  --override-tensor EXPR
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


def write_oom_llama_bench(path: Path) -> None:
    """Write a probeable llama-bench that fails every benchmark with OOM."""
    write_fake_llama_bench(path)
    script = path.read_text(encoding="utf-8")
    marker = 'repetitions = int(option("--repetitions", "3"))'
    script = script.replace(
        marker,
        'print("CUDA error: out of memory", file=sys.stderr)\n'
        "raise SystemExit(1)\n"
        + marker,
    )
    path.write_text(script, encoding="utf-8")


def write_fake_fit_params(path: Path, *, fail: bool = False) -> None:
    """Write a fake full-context fit tool with deterministic placement output."""
    script = f'''#!/usr/bin/env python3
import sys

HELP = """usage: llama-fit-params [options]
  --model PATH
  --ctx-size N
  --batch-size N
  --ubatch-size N
  --cache-type-k TYPE
  --cache-type-v TYPE
  --flash-attn MODE
  --load-mode MODE
  --lazy-mode MODE
  --fit-target N
  --fit-ctx N
  --split-mode MODE
  --main-gpu N
  --device LIST
  --kv-offload, --no-kv-offload
  --op-offload, --no-op-offload
  --no-host
  --repack, --no-repack
"""

if "--version" in sys.argv:
    print("version: 9998 (fedcba987)")
    raise SystemExit(0)
if "--help" in sys.argv or "-h" in sys.argv:
    print(HELP)
    raise SystemExit(0)
if {fail!r}:
    print("failed to fit CLI arguments to free memory", file=sys.stderr)
    raise SystemExit(1)

index = sys.argv.index("--ctx-size")
ctx = int(sys.argv[index + 1])
print("llama_params_fit: successfully fit params to free device memory")
print(f'-c {{ctx}} -ngl 42 -ts 3,1 -ot "blk\\.12\\.ffn_.*=CPU"')
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def register_fake_binary(database: Database, binary: Path) -> str:
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


def register_fake_bench(database: Database, binary: Path) -> str:
    """Backward-compatible test helper name used by CLI tests."""
    return register_fake_binary(database, binary)


def test_reference_experiment_runs_five_then_resumes_remaining_39(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "benchmarks.db")
    experiment_id = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, experiment_id)

    bench = tmp_path / "llama-bench"
    fit = tmp_path / "llama-fit-params"
    write_fake_llama_bench(bench)
    write_fake_fit_params(fit)
    binary_id = register_fake_binary(database, bench)
    fit_binary_id = register_fake_binary(database, fit)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake-model")

    executor = ExperimentExecutor(
        database,
        telemetry_provider_factory=LoadedTelemetryProvider,
    )
    first = executor.execute(
        experiment_id,
        binary_id=binary_id,
        fit_binary_id=fit_binary_id,
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
        assert connection.execute(
            "SELECT COUNT(*) FROM placement_attempt WHERE status = 'completed'"
        ).fetchone()[0] == 2

    second = executor.execute(
        experiment_id,
        binary_id=binary_id,
        fit_binary_id=fit_binary_id,
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
        placement_count = connection.execute(
            "SELECT COUNT(*) FROM resolved_placement"
        ).fetchone()[0]
        fit_count = connection.execute(
            "SELECT COUNT(*) FROM placement_attempt WHERE status = 'completed'"
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
        assert placement_count == 11
        assert fit_count == 11
        assert max_successes_per_case == 1


def test_one_candidate_is_fit_once_for_all_four_workloads(tmp_path: Path) -> None:
    database = Database(tmp_path / "fit-once.db")
    experiment_id = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, experiment_id)

    bench = tmp_path / "llama-bench"
    fit = tmp_path / "llama-fit-params"
    write_fake_llama_bench(bench)
    write_fake_fit_params(fit)
    bench_id = register_fake_binary(database, bench)
    fit_id = register_fake_binary(database, fit)
    model = tmp_path / "flash.gguf"
    model.write_bytes(b"flash")

    summary = ExperimentExecutor(database).execute(
        experiment_id,
        binary_id=bench_id,
        fit_binary_id=fit_id,
        model_path=model,
        limit=4,
    )

    assert summary.completed == 4
    with database.session() as connection:
        attempts = connection.execute(
            "SELECT COUNT(*) FROM placement_attempt WHERE status = 'completed'"
        ).fetchone()[0]
        placement_ids = {
            row[0]
            for row in connection.execute(
                """
                SELECT placement_id
                FROM benchmark_case
                WHERE experiment_id = ?
                ORDER BY ordinal
                LIMIT 4
                """,
                (experiment_id,),
            ).fetchall()
        }
        argv_rows = connection.execute(
            """
            SELECT benchmark_run.argv_json
            FROM benchmark_run
            JOIN benchmark_case
              ON benchmark_case.id = benchmark_run.benchmark_case_id
            WHERE benchmark_case.experiment_id = ?
              AND benchmark_run.status = 'completed'
            ORDER BY benchmark_case.ordinal
            LIMIT 4
            """,
            (experiment_id,),
        ).fetchall()

    assert attempts == 1
    assert len(placement_ids) == 1
    assert None not in placement_ids
    for row in argv_rows:
        argv_json = str(row[0])
        assert "--n-gpu-layers" in argv_json
        assert '"42"' in argv_json
        assert "--tensor-split" in argv_json
        assert "--override-tensor" in argv_json
        assert "--fit-target" not in argv_json


def test_fixed_placement_reuses_existing_resolution_without_fit(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "fixed.db")
    source_experiment = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, source_experiment)

    bench = tmp_path / "llama-bench"
    fit = tmp_path / "llama-fit-params"
    write_fake_llama_bench(bench)
    write_fake_fit_params(fit)
    bench_id = register_fake_binary(database, bench)
    fit_id = register_fake_binary(database, fit)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    first = ExperimentExecutor(database).execute(
        source_experiment,
        binary_id=bench_id,
        fit_binary_id=fit_id,
        model_path=model,
        limit=1,
    )
    assert first.completed == 1

    with database.session() as connection:
        placement_id = str(
            connection.execute("SELECT id FROM resolved_placement LIMIT 1").fetchone()[0]
        )
        source_definition = ExperimentRepository(connection).get_definition(
            source_experiment
        )
        assert source_definition is not None
        fixed_experiment = ExperimentRepository(connection).create(
            ExperimentDefinition(
                name="fixed placement reuse",
                base_candidate_id=source_definition.base_candidate_id,
                search_space_id=source_definition.search_space_id,
                workload_suite_id=source_definition.workload_suite_id,
                measurement_policy_id=source_definition.measurement_policy_id,
                placement_policy=FixedPlacementPolicy(placement_id=placement_id),
                baseline=source_definition.baseline,
            )
        )
        plan_experiment(connection, fixed_experiment)
        attempts_before = connection.execute(
            "SELECT COUNT(*) FROM placement_attempt"
        ).fetchone()[0]

    second = ExperimentExecutor(database).execute(
        fixed_experiment,
        binary_id=bench_id,
        model_path=model,
        limit=1,
    )

    assert second.completed == 1
    with database.session() as connection:
        attempts_after = connection.execute(
            "SELECT COUNT(*) FROM placement_attempt"
        ).fetchone()[0]
        bound = connection.execute(
            """
            SELECT placement_id FROM benchmark_case
            WHERE experiment_id = ?
            ORDER BY ordinal
            LIMIT 1
            """,
            (fixed_experiment,),
        ).fetchone()[0]

    assert attempts_after == attempts_before
    assert bound == placement_id


def test_fit_failure_is_persisted_and_candidate_is_retryable(tmp_path: Path) -> None:
    database = Database(tmp_path / "fit-failed.db")
    experiment_id = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, experiment_id)

    bench = tmp_path / "llama-bench"
    fit = tmp_path / "llama-fit-params"
    write_fake_llama_bench(bench)
    write_fake_fit_params(fit, fail=True)
    bench_id = register_fake_binary(database, bench)
    fit_id = register_fake_binary(database, fit)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    summary = ExperimentExecutor(database).execute(
        experiment_id,
        binary_id=bench_id,
        fit_binary_id=fit_id,
        model_path=model,
        limit=4,
    )

    assert summary.completed == 0
    assert summary.failed == 4
    assert summary.remaining == 44

    with database.session() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM placement_attempt WHERE status = 'fit_failed'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM benchmark_run"
        ).fetchone()[0] == 0
        statuses = {
            row[0]
            for row in connection.execute(
                """
                SELECT status FROM benchmark_case
                WHERE experiment_id = ?
                ORDER BY ordinal
                LIMIT 4
                """,
                (experiment_id,),
            ).fetchall()
        }
        assert statuses == {"fit_failed"}


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


def test_oom_attempt_remains_queryable_and_retryable(tmp_path: Path) -> None:
    database = Database(tmp_path / "oom.db")
    experiment_id = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, experiment_id)

    bench = tmp_path / "llama-bench"
    fit = tmp_path / "llama-fit-params"
    write_oom_llama_bench(bench)
    write_fake_fit_params(fit)
    bench_id = register_fake_binary(database, bench)
    fit_id = register_fake_binary(database, fit)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    summary = ExperimentExecutor(database).execute(
        experiment_id,
        binary_id=bench_id,
        fit_binary_id=fit_id,
        model_path=model,
        limit=1,
    )

    assert summary.completed == 0
    assert summary.failed == 1
    assert summary.remaining == 44
    with database.session() as connection:
        run = connection.execute(
            """
            SELECT benchmark_run.status, benchmark_run.stderr, benchmark_case.status
            FROM benchmark_run
            JOIN benchmark_case
              ON benchmark_case.id = benchmark_run.benchmark_case_id
            WHERE benchmark_case.experiment_id = ?
            ORDER BY benchmark_run.started_at
            LIMIT 1
            """,
            (experiment_id,),
        ).fetchone()
        assert run is not None
        assert run[0] == "oom"
        assert "out of memory" in str(run[1]).lower()
        assert run[2] == "oom"


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



class LoadedTelemetryProvider:
    """Deterministic provider that simulates substantial external CPU load."""

    logical_cpu_count = 16

    def __init__(self) -> None:
        self.timestamp = 0

    def sample(
        self,
        *,
        pid: int | None,
        phase: TelemetryPhase,
    ) -> TelemetrySample:
        self.timestamp += 1
        during = phase == "during"
        return TelemetrySample(
            timestamp_ns=self.timestamp,
            phase=phase,
            cpu_system_pct=80.0 if during else 5.0,
            cpu_user_pct=70.0 if during else 3.0,
            cpu_system_mode_pct=10.0 if during else 2.0,
            cpu_iowait_pct=0.0,
            process_cpu_pct_normalized=40.0 if during else None,
            process_cpu_pct_raw=640.0 if during else None,
            process_user_time_ns=1_000_000 if during else None,
            process_system_time_ns=100_000 if during else None,
            process_threads=8 if during else None,
            cpu_freq_avg_hz=4_000_000_000,
            cpu_freq_min_hz=3_900_000_000,
            cpu_freq_max_hz=4_100_000_000,
            cpu_temperature_c=65.0,
            load_avg_1m=8.0,
            load_avg_5m=4.0,
            ram_used_bytes=8_000_000_000,
            ram_available_bytes=24_000_000_000,
            swap_used_bytes=0,
            process_rss_bytes=2_000_000_000 if during else None,
            gpus=(
                GpuTelemetrySample(
                    device="0000:01:00.0",
                    utilization_pct=90.0 if during else 0.0,
                    vram_used_bytes=12_000_000_000 if during else 1_000_000_000,
                    vram_total_bytes=24_000_000_000,
                    temperature_c=70.0,
                    power_w=250.0 if during else 40.0,
                ),
            ),
            cpu_per_core_pct=(80.0,) * 16,
        )


def test_completed_run_persists_cpu_gpu_telemetry_and_external_load_quality(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "telemetry.db")
    experiment_id = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, experiment_id)

    bench = tmp_path / "llama-bench"
    fit = tmp_path / "llama-fit-params"
    write_fake_llama_bench(bench)
    write_fake_fit_params(fit)
    bench_id = register_fake_binary(database, bench)
    fit_id = register_fake_binary(database, fit)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    summary = ExperimentExecutor(
        database,
        telemetry_provider_factory=LoadedTelemetryProvider,
    ).execute(
        experiment_id,
        binary_id=bench_id,
        fit_binary_id=fit_id,
        model_path=model,
        limit=1,
        telemetry_interval_seconds=0.5,
    )

    assert summary.completed == 1
    with database.session() as connection:
        run_id = str(
            connection.execute(
                "SELECT id FROM benchmark_run WHERE status = 'completed' LIMIT 1"
            ).fetchone()[0]
        )
        run = BenchmarkRunRepository(connection).get(run_id)
        assert run is not None
        assert run.status == "completed"
        assert run.quality == "external_cpu_load"

        rows = connection.execute(
            """
            SELECT phase.value, telemetry_sample.process_cpu_pct_normalized,
                   telemetry_sample.cpu_system_pct, telemetry_sample.gpu_json
            FROM telemetry_sample
            JOIN json_each(telemetry_sample.extra_json, '$.phase') AS phase
            WHERE telemetry_sample.run_id = ?
            ORDER BY telemetry_sample.timestamp_ns
            """,
            (run_id,),
        ).fetchall()
        metrics = BenchmarkRunRepository(connection).metrics(run_id)

    assert rows[0][0] == "before"
    assert rows[-1][0] == "after"
    during_rows = [row for row in rows if row[0] == "during"]
    assert during_rows
    assert all(float(row[1]) == 40.0 for row in during_rows)
    assert all(float(row[2]) == 80.0 for row in during_rows)
    assert all("0000:01:00.0" in str(row[3]) for row in during_rows)
    assert metrics["telemetry.process_cpu_peak_pct_normalized"] == 40.0
    assert metrics["telemetry.gpu_utilization_peak_pct"] == 90.0
    assert metrics["telemetry.process_rss_peak_bytes"] == 2_000_000_000
