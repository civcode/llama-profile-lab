"""Synthetic completed reference observations for analysis tests."""

from __future__ import annotations

from pathlib import Path

from llama_profile_lab.db import (
    BenchmarkCaseRepository,
    BenchmarkRunRepository,
    CandidateRepository,
    Database,
    EnvironmentRepository,
    ExperimentRepository,
    WorkloadCaseRepository,
)
from llama_profile_lab.planning import plan_experiment
from tests.test_planning_persistence import seed_reference_experiment


def seed_analysis_experiment(path: Path) -> tuple[Database, str]:
    database = Database(path)
    experiment_id = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, experiment_id)

        host_id = EnvironmentRepository(connection).put_host(
            hostname="analysis-host",
            hardware_fingerprint="analysis-hardware",
            cpu={"logical_cpu_count": 16},
            ram_bytes=64_000_000_000,
            gpus=[{"pci_address": "0000:01:00.0"}],
            os_info={"system": "Linux"},
        )
        binary_id = EnvironmentRepository(connection).put_binary(
            sha256="a" * 64,
            kind="llama-bench",
            path="/synthetic/llama-bench",
            size_bytes=1,
            mtime_ns=1,
        )
        definition = ExperimentRepository(connection).get_definition(experiment_id)
        assert definition is not None

        candidates = CandidateRepository(connection)
        workloads = WorkloadCaseRepository(connection)
        runs = BenchmarkRunRepository(connection)
        cases = BenchmarkCaseRepository(connection)
        rows = connection.execute(
            """
            SELECT bc.id, bc.candidate_id, bc.workload_case_id, ew.suite_case_index
            FROM benchmark_case AS bc
            JOIN experiment_workload AS ew
              ON ew.experiment_id = bc.experiment_id
             AND ew.candidate_id = bc.candidate_id
             AND ew.workload_case_id = bc.workload_case_id
            WHERE bc.experiment_id = ?
            ORDER BY bc.ordinal
            """,
            (experiment_id,),
        ).fetchall()

        for row in rows:
            candidate = candidates.get(str(row["candidate_id"]))
            workload = workloads.get(str(row["workload_case_id"]))
            assert candidate is not None
            assert workload is not None
            batch = candidate.compute.batch_size
            ubatch = candidate.compute.ubatch_size
            suite_index = int(row["suite_case_index"])

            if workload.kind == "microbench-prefill":
                scale = 0.75 if workload.prompt_tokens == 2048 else 1.0
                throughput = batch / 64.0 * scale
            else:
                depth_penalty = workload.depth_tokens / 16_384.0
                throughput = 300.0 - batch / 128.0 - ubatch / 64.0 - depth_penalty

            cpu = batch / 4096.0 + ubatch / 128.0
            rss = batch * 1_000_000 + ubatch * 2_000_000
            gpu_util = min(99.0, 35.0 + batch / 256.0)
            run_id = runs.create(
                benchmark_case_id=str(row["id"]),
                host_id=host_id,
                binary_id=binary_id,
                measurement_policy_id=definition.measurement_policy_id,
                argv=("llama-bench", f"synthetic-{suite_index}"),
                environment={},
            )
            samples = (
                (1_000_000_000, throughput * 0.98),
                (1_000_000_000, throughput),
                (1_000_000_000, throughput * 1.02),
            )
            runs.add_samples(run_id, samples)
            runs.add_metrics(
                run_id,
                {
                    "telemetry.process_cpu_avg_pct_normalized": cpu,
                    "telemetry.process_cpu_peak_pct_normalized": cpu * 1.1,
                    "telemetry.cpu_system_avg_pct": cpu + 5.0,
                    "telemetry.cpu_system_peak_pct": cpu + 8.0,
                    "telemetry.process_user_time_ns": 800_000_000,
                    "telemetry.process_system_time_ns": 200_000_000,
                    "telemetry.ram_used_peak_bytes": rss + 4_000_000_000,
                    "telemetry.process_rss_peak_bytes": rss,
                    "telemetry.gpu_utilization_avg_pct": gpu_util,
                    "telemetry.gpu_utilization_peak_pct": min(100.0, gpu_util + 5.0),
                    "telemetry.gpu_vram_used_peak_bytes": rss * 2,
                    "telemetry.gpu_temperature_peak_c": 70.0 + batch / 8192.0,
                    "telemetry.gpu_power_avg_w": 150.0 + batch / 128.0,
                },
            )
            runs.set_quality(run_id, quality="clean", details={"reasons": []})
            runs.finish(
                run_id,
                status="completed",
                duration_ns=1_000_000_000,
                exit_code=0,
            )
            cases.set_status(str(row["id"]), "completed")

    return database, experiment_id


def candidate_id_for(
    database: Database,
    experiment_id: str,
    *,
    batch: int,
    ubatch: int,
) -> str:
    with database.session() as connection:
        candidates = CandidateRepository(connection)
        rows = connection.execute(
            """
            SELECT candidate_id
            FROM experiment_candidate
            WHERE experiment_id = ?
            ORDER BY ordinal
            """,
            (experiment_id,),
        ).fetchall()
        for row in rows:
            candidate_id = str(row["candidate_id"])
            candidate = candidates.get(candidate_id)
            assert candidate is not None
            if (
                candidate.compute.batch_size == batch
                and candidate.compute.ubatch_size == ubatch
            ):
                return candidate_id
    raise AssertionError(f"candidate not found for batch={batch}, ubatch={ubatch}")
