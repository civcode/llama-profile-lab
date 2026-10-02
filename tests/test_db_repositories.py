"""Persistence round-trip and append-only run tests."""

from pathlib import Path

import pytest
from sqlite3 import IntegrityError

from llama_profile_lab.db import (
    BenchmarkCaseRepository,
    BenchmarkRunRepository,
    CandidateRepository,
    Database,
    EnvironmentRepository,
    ExperimentRepository,
    MeasurementPolicyRepository,
    SearchSpaceRepository,
    WorkloadCaseRepository,
    WorkloadSuiteRepository,
)
from llama_profile_lab.domain import (
    AbsoluteDepth,
    Candidate,
    ComputeConfig,
    ContextConfig,
    ExperimentDefinition,
    FitConfig,
    MeasurementPolicy,
    ModelSelection,
    PlacementConfig,
    PrefillSuiteCase,
    PrefillWorkloadCase,
    SearchDimension,
    SearchSpace,
    WorkloadSuite,
)


def make_candidate() -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id="model:sha256:target"),
        context=ContextConfig(
            size=131072,
            cache_type_k="f16",
            cache_type_v="f16",
        ),
        compute=ComputeConfig(
            flash_attn="on",
            batch_size=4096,
            ubatch_size=2048,
            threads=16,
            load_mode="mmap",
            lazy_mode="on",
        ),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=256, min_context=4096),
        ),
    )


def test_immutable_domain_objects_round_trip_across_reopen(tmp_path: Path) -> None:
    database = Database(tmp_path / "benchmarks.db")
    candidate = make_candidate()
    search_space = SearchSpace(
        dimensions=(
            SearchDimension(path="compute.batch_size", values=(2048, 4096)),
            SearchDimension(path="compute.ubatch_size", values=(512, 1024)),
        ),
        constraints=("compute.ubatch_size <= compute.batch_size",),
    )
    suite = WorkloadSuite(
        id="screen-v1",
        cases=(
            PrefillSuiteCase(
                label="pp-2k",
                prompt_tokens=2048,
                depth=AbsoluteDepth(tokens=0),
            ),
        ),
    )
    workload = PrefillWorkloadCase(prompt_tokens=2048, depth_tokens=0)
    policy = MeasurementPolicy(repetitions=3)

    with database.session() as connection:
        candidate_repo = CandidateRepository(connection)
        search_repo = SearchSpaceRepository(connection)
        suite_repo = WorkloadSuiteRepository(connection)
        workload_repo = WorkloadCaseRepository(connection)
        policy_repo = MeasurementPolicyRepository(connection)
        experiment_repo = ExperimentRepository(connection)

        candidate_id = candidate_repo.put(candidate)
        assert candidate_repo.put(candidate) == candidate_id
        search_space_id = search_repo.put(search_space)
        suite_id = suite_repo.put(suite)
        workload_id = workload_repo.put(workload)
        policy_id = policy_repo.put(policy)

        experiment = ExperimentDefinition(
            name="Persistence round trip",
            base_candidate_id=candidate_id,
            search_space_id=search_space_id,
            workload_suite_id=suite_id,
            measurement_policy_id=policy_id,
        )
        experiment_id = experiment_repo.create(experiment)

    with database.session() as connection:
        assert CandidateRepository(connection).get(candidate_id) == candidate
        assert SearchSpaceRepository(connection).get(search_space_id) == search_space
        assert WorkloadSuiteRepository(connection).get(suite_id) == suite
        assert WorkloadCaseRepository(connection).get(workload_id) == workload
        assert MeasurementPolicyRepository(connection).get(policy_id) == policy
        assert ExperimentRepository(connection).get_definition(experiment_id) == experiment

        record = ExperimentRepository(connection).get(experiment_id)
        assert record is not None
        assert record.status == "draft"
        assert record.base_candidate_id == candidate_id


def test_foreign_keys_are_enforced_for_experiments(tmp_path: Path) -> None:
    database = Database(tmp_path / "foreign-keys.db")
    with database.session() as connection:
        with pytest.raises(IntegrityError):
            ExperimentRepository(connection).create(
                ExperimentDefinition(
                    name="invalid",
                    base_candidate_id="cand_missing",
                    search_space_id="space_missing",
                    workload_suite_id="suite_missing",
                    measurement_policy_id="measure_missing",
                )
            )


def test_benchmark_runs_are_distinct_and_finalization_is_one_way(tmp_path: Path) -> None:
    database = Database(tmp_path / "runs.db")
    candidate = make_candidate()
    search_space = SearchSpace(
        dimensions=(SearchDimension(path="compute.batch_size", values=(4096,)),)
    )
    suite = WorkloadSuite(
        id="one-case",
        cases=(
            PrefillSuiteCase(
                prompt_tokens=2048,
                depth=AbsoluteDepth(tokens=0),
            ),
        ),
    )
    workload = PrefillWorkloadCase(prompt_tokens=2048, depth_tokens=0)
    policy = MeasurementPolicy(repetitions=3)

    with database.session() as connection:
        candidate_id = CandidateRepository(connection).put(candidate)
        search_id = SearchSpaceRepository(connection).put(search_space)
        suite_id = WorkloadSuiteRepository(connection).put(suite)
        workload_id = WorkloadCaseRepository(connection).put(workload)
        policy_id = MeasurementPolicyRepository(connection).put(policy)
        experiment_id = ExperimentRepository(connection).create(
            ExperimentDefinition(
                name="run persistence",
                base_candidate_id=candidate_id,
                search_space_id=search_id,
                workload_suite_id=suite_id,
                measurement_policy_id=policy_id,
            ),
            status="planned",
        )
        case_id = BenchmarkCaseRepository(connection).put_planned(
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            workload_case_id=workload_id,
            ordinal=0,
        )

        environment_repo = EnvironmentRepository(connection)
        host_id = environment_repo.put_host(
            hostname="test-host",
            hardware_fingerprint="cpu:test|gpu:test",
            cpu={"model": "test"},
            ram_bytes=64 * 1024**3,
            gpus=[{"name": "test-gpu"}],
            os_info={"kernel": "test"},
        )
        binary_id = environment_repo.put_binary(
            sha256="a" * 64,
            kind="llama-bench",
            path="/opt/llama-bench",
            size_bytes=1234,
            mtime_ns=5678,
        )

        run_repo = BenchmarkRunRepository(connection)
        first_run = run_repo.create(
            benchmark_case_id=case_id,
            host_id=host_id,
            binary_id=binary_id,
            measurement_policy_id=policy_id,
            argv=("llama-bench", "--model", "model.gguf"),
            environment={"PATH": "/usr/bin"},
        )
        run_repo.finish(
            first_run,
            status="completed",
            duration_ns=1_000_000,
            exit_code=0,
            raw_result={"avg_ts": 42.0},
        )

        with pytest.raises(ValueError, match="already been finalized"):
            run_repo.finish(
                first_run,
                status="completed",
                duration_ns=1_000_000,
                exit_code=0,
            )

        second_run = run_repo.create(
            benchmark_case_id=case_id,
            host_id=host_id,
            binary_id=binary_id,
            measurement_policy_id=policy_id,
            argv=("llama-bench", "--model", "model.gguf"),
            environment={"PATH": "/usr/bin"},
        )

        assert first_run != second_run
        assert run_repo.get(first_run).status == "completed"
        assert run_repo.get(second_run).status == "running"
