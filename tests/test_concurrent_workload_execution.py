"""Deterministic integration tests for concurrent deployment workloads."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Thread

import pytest

from llama_profile_lab.db import (
    CandidateRepository,
    ConcurrentWorkloadRepository,
    Database,
    DeploymentCandidateRepository,
    DeploymentPlacementRepository,
    DeploymentRunRepository,
    EnvironmentRepository,
    PlacementRepository,
    WorkloadSuiteRepository,
)
from llama_profile_lab.domain import (
    AbsoluteDepth,
    Candidate,
    ComputeConfig,
    ConcurrentMemberResult,
    ConcurrentTokenEvent,
    ContextConfig,
    DecodeSuiteCase,
    DeploymentCandidate,
    DeploymentInstancePlacement,
    DeploymentPlacement,
    DeploymentWorkloadMix,
    FitConfig,
    HostResourcePolicy,
    ModelInstanceCandidate,
    ModelSelection,
    PlacementConfig,
    PrefillSuiteCase,
    ResolvedPlacement,
    WorkloadSuite,
)
from llama_profile_lab.execution import (
    ConcurrentDeploymentExecutor,
    DeploymentExecutionMember,
    DeploymentExecutionSummary,
    DeploymentResidentActionError,
    DeploymentServerInput,
    ResidentDeployment,
    ResidentDeploymentMember,
    StandaloneBaselineInput,
)
from llama_profile_lab.llama import sha256_file


SECOND = 1_000_000_000


class FakeResidencyExecutor:
    """Persist one deployment_run and invoke the real M6 resident action."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def execute(
        self,
        deployment_placement_id,
        inputs,
        *,
        host,
        readiness_timeout_seconds,
        cancel_event,
        resident_action,
        **kwargs,
    ) -> DeploymentExecutionSummary:
        del inputs, host, readiness_timeout_seconds, kwargs
        with self.database.session() as connection:
            placement = DeploymentPlacementRepository(connection).get(
                deployment_placement_id
            )
            assert placement is not None
            runs = DeploymentRunRepository(connection)
            run_id = runs.create(
                deployment_candidate_id=placement.deployment_candidate_id,
                deployment_placement_id=deployment_placement_id,
                status="ready",
            )

        resident = ResidentDeployment(
            run_id=run_id,
            deployment_candidate_id=placement.deployment_candidate_id,
            deployment_placement_id=deployment_placement_id,
            members=(
                ResidentDeploymentMember(
                    instance_id="a",
                    endpoint="http://127.0.0.1:41001",
                    pid=101,
                ),
                ResidentDeploymentMember(
                    instance_id="b",
                    endpoint="http://127.0.0.1:41002",
                    pid=102,
                ),
            ),
        )
        try:
            output = resident_action(resident, cancel_event)
        except DeploymentResidentActionError as exc:
            with self.database.session() as connection:
                DeploymentRunRepository(connection).finish(
                    run_id,
                    status="cancelled" if exc.cancelled else "failed",
                    duration_ns=1,
                    failure_kind=(
                        None if exc.cancelled else exc.failure_kind
                    ),
                    quality_details=(
                        {"resident_action": {}}
                        if exc.cancelled
                        else None
                    ),
                    failure_details=(
                        None if exc.cancelled else {"error": str(exc)}
                    ),
                )
            raise

        with self.database.session() as connection:
            DeploymentRunRepository(connection).finish(
                run_id,
                status="completed",
                duration_ns=1,
                quality_details={"resident_action": dict(output or {})},
            )
        members = (
            DeploymentExecutionMember(
                instance_id="a",
                endpoint="http://127.0.0.1:41001",
                pid=101,
                ready_at="2026-10-05T12:00:00Z",
                exit_code=0,
                forced_kill=False,
            ),
            DeploymentExecutionMember(
                instance_id="b",
                endpoint="http://127.0.0.1:41002",
                pid=102,
                ready_at="2026-10-05T12:00:00Z",
                exit_code=0,
                forced_kill=False,
            ),
        )
        return DeploymentExecutionSummary(
            run_id=run_id,
            deployment_candidate_id=placement.deployment_candidate_id,
            deployment_placement_id=deployment_placement_id,
            status="completed",
            members=members,
            runtime_memory={},
        )


class DeterministicClient:
    def __init__(
        self,
        *,
        status_by_instance: dict[str, str] | None = None,
        correctness_by_instance: dict[str, bool] | None = None,
        duration_by_instance: dict[str, int] | None = None,
        start_offset_by_instance: dict[str, int] | None = None,
        block_until_cancel: bool = False,
    ) -> None:
        self.status_by_instance = status_by_instance or {}
        self.correctness_by_instance = correctness_by_instance or {}
        self.duration_by_instance = duration_by_instance or {}
        self.start_offset_by_instance = start_offset_by_instance or {}
        self.block_until_cancel = block_until_cancel

    def prepare(self, endpoint, workload, *, cancel_event=None):
        del endpoint, cancel_event
        return _PreparedDeterministicClient(
            workload=workload,
            ready_ns=time.monotonic_ns(),
            parent=self,
        )


class _PreparedDeterministicClient:
    def __init__(self, *, workload, ready_ns: int, parent) -> None:
        self.workload = workload
        self.client_ready_ns = ready_ns
        self.parent = parent

    def run(self, *, barrier_release_ns: int, cancel_event=None):
        if self.parent.block_until_cancel:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    return _result(
                        self.workload,
                        barrier_release_ns,
                        status="cancelled",
                        correctness=False,
                    )
                time.sleep(0.01)
            pytest.fail("fake concurrent client was not cancelled")

        status = self.parent.status_by_instance.get(
            self.workload.instance_id,
            "completed",
        )
        correctness = self.parent.correctness_by_instance.get(
            self.workload.instance_id,
            True,
        )
        return _result(
            self.workload,
            barrier_release_ns,
            status=status,
            correctness=correctness,
            duration_ns=self.parent.duration_by_instance.get(
                self.workload.instance_id,
                2 * SECOND,
            ),
            start_offset_ns=self.parent.start_offset_by_instance.get(
                self.workload.instance_id,
                0,
            ),
        )


def _result(
    workload,
    release_ns: int,
    *,
    status: str,
    correctness: bool,
    duration_ns: int = 2 * SECOND,
    start_offset_ns: int = 0,
) -> ConcurrentMemberResult:
    start = release_ns + start_offset_ns
    finish = start + duration_ns
    failure = None if status == "completed" else f"synthetic {status}"

    if workload.mode == "prefill":
        total = workload.prompt_tokens
        events = (
            ConcurrentTokenEvent(
                kind="prefill",
                timestamp_ns=start + duration_ns // 2,
                cumulative_tokens=total // 2,
            ),
            ConcurrentTokenEvent(
                kind="prefill",
                timestamp_ns=finish,
                cumulative_tokens=total,
            ),
        )
        return ConcurrentMemberResult(
            instance_id=workload.instance_id,
            mode="prefill",
            status="invalid" if not correctness else status,
            client_ready_ns=min(start, release_ns),
            barrier_release_ns=release_ns,
            first_request_ns=start,
            finished_ns=finish,
            prompt_tokens=total if status == "completed" else 0,
            native_prompt_tps=(
                total / (duration_ns / SECOND)
                if status == "completed"
                else None
            ),
            latency_ms=duration_ns / 1_000_000,
            token_events=events if status == "completed" else (),
            correctness_valid=correctness and status == "completed",
            failure=failure,
            raw={"fixture": "prefill"},
        )

    total = workload.generate_tokens
    events = (
        ConcurrentTokenEvent(
            kind="decode",
            timestamp_ns=start + duration_ns // 2,
            cumulative_tokens=total // 2,
        ),
        ConcurrentTokenEvent(
            kind="decode",
            timestamp_ns=finish,
            cumulative_tokens=total,
        ),
    )
    return ConcurrentMemberResult(
        instance_id=workload.instance_id,
        mode="decode",
        status="invalid" if not correctness else status,
        client_ready_ns=min(start, release_ns),
        barrier_release_ns=release_ns,
        first_request_ns=start,
        first_token_ns=events[0].timestamp_ns,
        last_token_ns=events[-1].timestamp_ns,
        finished_ns=finish,
        decode_tokens=total if status == "completed" else 0,
        native_decode_tps=(
            total / (duration_ns / SECOND)
            if status == "completed"
            else None
        ),
        latency_ms=duration_ns / 1_000_000,
        token_events=events if status == "completed" else (),
        correctness_valid=correctness and status == "completed",
        failure=failure,
        raw={"fixture": "decode"},
    )


def _candidate(model_id: str) -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id=model_id),
        context=ContextConfig(size=4096),
        compute=ComputeConfig(batch_size=512, ubatch_size=128),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=0, min_context=1024),
        ),
    )


def _seed(
    tmp_path: Path,
) -> tuple[Database, str, tuple[DeploymentServerInput, ...]]:
    database = Database(tmp_path / "concurrent.db")
    server = tmp_path / "llama-server"
    server.write_bytes(b"fake server")
    model_a = tmp_path / "a.gguf"
    model_b = tmp_path / "b.gguf"
    model_a.write_bytes(b"a")
    model_b.write_bytes(b"b")

    with database.session() as connection:
        candidates = CandidateRepository(connection)
        cand_a = candidates.put(_candidate("model:a"))
        cand_b = candidates.put(_candidate("model:b"))
        suite_id = WorkloadSuiteRepository(connection).put(
            WorkloadSuite(
                id="concurrent-suite",
                cases=(
                    PrefillSuiteCase(
                        prompt_tokens=100,
                        depth=AbsoluteDepth(tokens=128),
                    ),
                    DecodeSuiteCase(
                        generate_tokens=20,
                        depth=AbsoluteDepth(tokens=128),
                    ),
                ),
            )
        )
        environment = EnvironmentRepository(connection)
        binary_id = environment.put_binary(
            sha256=sha256_file(server),
            kind="llama-server",
            path=str(server),
            size_bytes=server.stat().st_size,
            mtime_ns=server.stat().st_mtime_ns,
            capabilities={},
        )
        host_id = environment.put_host(
            hostname="m6-host",
            hardware_fingerprint="m6-host-fingerprint",
            cpu={},
            ram_bytes=1,
            gpus=[],
            os_info={},
        )
        deployment = DeploymentCandidate(
            instances=(
                ModelInstanceCandidate(
                    instance_id="a",
                    candidate_id=cand_a,
                    role="primary",
                    model_artifact_id="model:a",
                    binary_id=binary_id,
                    server_identity="a",
                ),
                ModelInstanceCandidate(
                    instance_id="b",
                    candidate_id=cand_b,
                    role="secondary",
                    model_artifact_id="model:b",
                    binary_id=binary_id,
                    server_identity="b",
                ),
            ),
            resource_policy=HostResourcePolicy(),
            workload_mix=DeploymentWorkloadMix(
                workload_suite_id=suite_id
            ),
        )
        deployment_id = DeploymentCandidateRepository(connection).put(
            deployment
        )
        placements = PlacementRepository(connection)
        place_a = placements.put_resolved(
            placement_hash="a" * 64,
            candidate_id=cand_a,
            host_id=host_id,
            binary_id=binary_id,
            fit_attempt_id=None,
            placement=ResolvedPlacement(
                production_context_size=4096,
                n_gpu_layers=1,
                devices=("CUDA0",),
            ),
            request={},
            argv=(),
            stdout="",
            stderr="",
            exit_code=0,
        )
        place_b = placements.put_resolved(
            placement_hash="b" * 64,
            candidate_id=cand_b,
            host_id=host_id,
            binary_id=binary_id,
            fit_attempt_id=None,
            placement=ResolvedPlacement(
                production_context_size=4096,
                n_gpu_layers=1,
                devices=("Vulkan0",),
            ),
            request={},
            argv=(),
            stdout="",
            stderr="",
            exit_code=0,
        )
        deployment_placement_id = DeploymentPlacementRepository(
            connection
        ).put(
            DeploymentPlacement(
                deployment_candidate_id=deployment_id,
                host_id=host_id,
                instance_placements=(
                    DeploymentInstancePlacement(
                        instance_id="a",
                        resolved_placement_id=place_a,
                    ),
                    DeploymentInstancePlacement(
                        instance_id="b",
                        resolved_placement_id=place_b,
                    ),
                ),
                feasibility="feasible",
            )
        )

    return (
        database,
        deployment_placement_id,
        (
            DeploymentServerInput("a", model_a),
            DeploymentServerInput("b", model_b),
        ),
    )


def _baselines() -> tuple[StandaloneBaselineInput, ...]:
    return (
        StandaloneBaselineInput("a", "prefill", 100, 0, 128, 100.0),
        StandaloneBaselineInput("b", "prefill", 100, 0, 128, 100.0),
        StandaloneBaselineInput("a", "decode", 0, 20, 128, 20.0),
        StandaloneBaselineInput("b", "decode", 0, 20, 128, 20.0),
    )


def _service(
    database: Database,
    client: DeterministicClient | None = None,
) -> ConcurrentDeploymentExecutor:
    return ConcurrentDeploymentExecutor(
        database,
        deployment_executor=FakeResidencyExecutor(database),
        client=client or DeterministicClient(),
    )


def test_executes_all_four_phases_with_retention_and_timing(
    tmp_path: Path,
) -> None:
    database, placement_id, inputs = _seed(tmp_path)

    summary = _service(database).execute(
        placement_id,
        inputs,
        standalone_baselines=_baselines(),
    )

    assert [item.phase for item in summary.phases] == [
        "dd",
        "pp",
        "pd",
        "dp",
    ]
    assert all(item.quality == "clean" for item in summary.phases)
    assert all(item.min_retention == pytest.approx(0.5) for item in summary.phases)

    by_phase = {item.phase: item for item in summary.phases}
    assert by_phase["dd"].combined_decode_tps == pytest.approx(20.0)
    assert by_phase["pp"].combined_prompt_tps == pytest.approx(100.0)
    assert by_phase["pd"].combined_prompt_tps == pytest.approx(50.0)
    assert by_phase["pd"].combined_decode_tps == pytest.approx(10.0)

    with database.session() as connection:
        rows = connection.execute(
            """
            SELECT id, phase, barrier_release_ns, min_retention
            FROM deployment_workload_run
            WHERE deployment_run_id = ?
            ORDER BY created_at, id
            """,
            (summary.deployment_run_id,),
        ).fetchall()
        assert len(rows) == 4
        member_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM deployment_workload_member AS m
            JOIN deployment_workload_run AS r
              ON r.id = m.deployment_workload_run_id
            WHERE r.deployment_run_id = ?
            """,
            (summary.deployment_run_id,),
        ).fetchone()[0]
        assert member_count == 8

        for row in rows:
            members = ConcurrentWorkloadRepository(connection).members(
                str(row["id"])
            )
            assert len(members) == 2
            assert all(
                item.first_request_ns == row["barrier_release_ns"]
                for item in members
            )
            assert all(item.retention == pytest.approx(0.5) for item in members)
            assert all(item.raw["fixture"] in {"prefill", "decode"} for item in members)


def test_missing_and_ambiguous_baselines_are_not_guessed(tmp_path: Path) -> None:
    database, placement_id, inputs = _seed(tmp_path)

    missing = _service(database).execute(placement_id, inputs)
    assert all(item.quality == "baseline_missing" for item in missing.phases)
    assert all(item.min_retention is None for item in missing.phases)

    database2, placement_id2, inputs2 = _seed(tmp_path / "ambiguous")
    duplicated = (*_baselines(), _baselines()[0])
    ambiguous = _service(database2).execute(
        placement_id2,
        inputs2,
        standalone_baselines=duplicated,
    )
    qualities = {item.quality for item in ambiguous.phases}
    assert "baseline_ambiguous" in qualities


@pytest.mark.parametrize(
    ("status", "expected_quality", "expected_kind"),
    (
        ("failed", "member_failed", "concurrent_workload_failed"),
        ("timeout", "member_timeout", "member_timeout"),
    ),
)
def test_member_failure_and_timeout_are_persisted(
    tmp_path: Path,
    status: str,
    expected_quality: str,
    expected_kind: str,
) -> None:
    database, placement_id, inputs = _seed(tmp_path)
    client = DeterministicClient(status_by_instance={"a": status})

    with pytest.raises(DeploymentResidentActionError) as captured:
        _service(database, client).execute(placement_id, inputs)

    assert captured.value.failure_kind == expected_kind
    with database.session() as connection:
        row = connection.execute(
            """
            SELECT id, quality
            FROM deployment_workload_run
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        assert row is not None
        assert row["quality"] == expected_quality


def test_output_validation_failure_is_persisted(tmp_path: Path) -> None:
    database, placement_id, inputs = _seed(tmp_path)
    client = DeterministicClient(correctness_by_instance={"b": False})

    with pytest.raises(DeploymentResidentActionError) as captured:
        _service(database, client).execute(placement_id, inputs)

    assert captured.value.failure_kind == "output_validation_failed"
    with database.session() as connection:
        row = connection.execute(
            """
            SELECT quality, correctness_valid
            FROM deployment_workload_run
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
    assert row is not None
    assert row["quality"] == "correctness_invalid"
    assert row["correctness_valid"] == 0


def test_cancellation_during_concurrent_phase(tmp_path: Path) -> None:
    database, placement_id, inputs = _seed(tmp_path)
    cancel = Event()
    client = DeterministicClient(block_until_cancel=True)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            _service(database, client).execute(
                placement_id,
                inputs,
                cancel_event=cancel,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=run)
    thread.start()
    time.sleep(0.1)
    cancel.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], DeploymentResidentActionError)
    assert errors[0].cancelled is True


def test_staggered_completion_uses_shared_overlap_only(tmp_path: Path) -> None:
    database, placement_id, inputs = _seed(tmp_path)
    client = DeterministicClient(
        duration_by_instance={"a": 2 * SECOND, "b": 4 * SECOND}
    )

    summary = _service(database, client).execute(
        placement_id,
        inputs,
        standalone_baselines=_baselines(),
    )

    dd = next(item for item in summary.phases if item.phase == "dd")
    assert dd.combined_decode_tps == pytest.approx(15.0)
