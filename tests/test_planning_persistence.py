"""End-to-end experiment planning and persistence tests."""

from pathlib import Path

from llama_profile_lab.db import (
    CandidateRepository,
    Database,
    ExperimentRepository,
    MeasurementPolicyRepository,
    SearchSpaceRepository,
    WorkloadSuiteRepository,
)
from llama_profile_lab.domain import (
    AbsoluteDepth,
    Candidate,
    ComputeConfig,
    ContextConfig,
    DecodeSuiteCase,
    ExperimentDefinition,
    FitConfig,
    FractionalDepth,
    MeasurementPolicy,
    ModelSelection,
    PlacementConfig,
    PrefillSuiteCase,
    SearchDimension,
    SearchSpace,
    WorkloadSuite,
)
from llama_profile_lab.planning import plan_experiment


def seed_reference_experiment(database: Database) -> str:
    base = Candidate(
        model=ModelSelection(target_model_id="model:sha256:flash"),
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
    search = SearchSpace(
        dimensions=(
            SearchDimension(
                path="compute.batch_size",
                values=(2048, 4096, 8192),
            ),
            SearchDimension(
                path="compute.ubatch_size",
                values=(512, 1024, 2048, 4096),
            ),
        ),
        constraints=("compute.ubatch_size <= compute.batch_size",),
    )
    suite = WorkloadSuite(
        id="batch-ubatch-screen-v1",
        cases=(
            PrefillSuiteCase(
                label="pp-2k",
                prompt_tokens=2048,
                depth=AbsoluteDepth(tokens=0),
            ),
            PrefillSuiteCase(
                label="pp-8k",
                prompt_tokens=8192,
                depth=AbsoluteDepth(tokens=0),
            ),
            DecodeSuiteCase(
                label="tg-short",
                generate_tokens=256,
                depth=AbsoluteDepth(tokens=4096),
            ),
            DecodeSuiteCase(
                label="tg-mid",
                generate_tokens=256,
                depth=FractionalDepth(value=0.5),
            ),
        ),
    )
    policy = MeasurementPolicy(repetitions=3)

    with database.session() as connection:
        base_id = CandidateRepository(connection).put(base)
        search_id = SearchSpaceRepository(connection).put(search)
        suite_id = WorkloadSuiteRepository(connection).put(suite)
        policy_id = MeasurementPolicyRepository(connection).put(policy)
        return ExperimentRepository(connection).create(
            ExperimentDefinition(
                name="Flash Next 128K batch/ubatch",
                base_candidate_id=base_id,
                search_space_id=search_id,
                workload_suite_id=suite_id,
                measurement_policy_id=policy_id,
            )
        )


def test_reference_experiment_persists_eleven_candidates_and_44_cases(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "benchmarks.db")
    experiment_id = seed_reference_experiment(database)

    with database.session() as connection:
        summary = plan_experiment(connection, experiment_id)

    assert summary.raw_combinations == 12
    assert summary.rejected_by_constraints == 1
    assert summary.candidate_count == 11
    assert summary.workloads_per_candidate == 4
    assert summary.benchmark_case_count == 44
    assert summary.unique_workload_count == 4

    with database.session() as connection:
        record = ExperimentRepository(connection).get(experiment_id)
        assert record is not None
        assert record.status == "planned"
        assert record.frozen_at is not None

        candidate_count = connection.execute(
            "SELECT COUNT(*) FROM experiment_candidate WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]
        workload_links = connection.execute(
            "SELECT COUNT(*) FROM experiment_workload WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]
        benchmark_cases = connection.execute(
            "SELECT COUNT(*) FROM benchmark_case WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]

        assert candidate_count == 11
        assert workload_links == 44
        assert benchmark_cases == 44

        tg_mid_depths = {
            row[0]
            for row in connection.execute(
                """
                SELECT workload_case.depth_tokens
                FROM experiment_workload
                JOIN workload_case
                  ON workload_case.id = experiment_workload.workload_case_id
                WHERE experiment_workload.experiment_id = ?
                  AND workload_case.kind = 'microbench-decode'
                  AND workload_case.depth_tokens != 4096
                """,
                (experiment_id,),
            ).fetchall()
        }
        assert tg_mid_depths == {65408}
