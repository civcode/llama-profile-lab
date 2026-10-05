"""Deployment analysis, interference, export, and Pareto tests."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from llama_profile_lab.analysis import (
    DeploymentAnalysisError,
    DeploymentAnalysisFilter,
    DeploymentAnalysisService,
    DeploymentMetricConstraint,
    DeploymentParetoObjective,
)
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
    ConcurrentWorkloadCase,
    ConcurrentWorkloadMemberSpec,
    ContextConfig,
    DecodeSuiteCase,
    DeploymentCandidate,
    DeploymentDeviceAllocation,
    DeploymentInstancePlacement,
    DeploymentPlacement,
    DeploymentWorkloadMix,
    FitConfig,
    GpuTelemetrySample,
    HostResourcePolicy,
    ModelInstanceCandidate,
    ModelSelection,
    PlacementConfig,
    PlacementDeviceMemory,
    PrefillSuiteCase,
    ResolvedPlacement,
    WorkloadSuite,
)
from llama_profile_lab.llama import sha256_file


def _candidate(model_id: str, context_size: int) -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id=model_id),
        context=ContextConfig(
            size=context_size,
            cache_type_k="f16",
            cache_type_v="f16",
        ),
        compute=ComputeConfig(
            flash_attn="on",
            batch_size=512,
            ubatch_size=128,
        ),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=0, min_context=1024),
        ),
    )


def _workload(phase: str, depth: int = 128) -> ConcurrentWorkloadCase:
    modes = {
        "dd": ("decode", "decode"),
        "pp": ("prefill", "prefill"),
        "pd": ("prefill", "decode"),
        "dp": ("decode", "prefill"),
    }[phase]
    members = []
    for ordinal, (instance_id, mode) in enumerate(
        zip(("qwen", "flash"), modes, strict=True)
    ):
        members.append(
            ConcurrentWorkloadMemberSpec(
                ordinal=ordinal,
                instance_id=instance_id,
                mode=mode,
                prompt_tokens=100 if mode == "prefill" else 0,
                generate_tokens=20 if mode == "decode" else 0,
                depth_tokens=depth,
            )
        )
    return ConcurrentWorkloadCase(
        phase=phase,
        members=tuple(members),
    )


def _result(
    spec: ConcurrentWorkloadMemberSpec,
    *,
    retention: float,
) -> ConcurrentMemberResult:
    standalone = 100.0 if spec.mode == "prefill" else 20.0
    native = standalone * retention
    events = (
        ConcurrentTokenEvent(
            kind=spec.mode,
            timestamp_ns=1_500_000_000,
            cumulative_tokens=(
                spec.prompt_tokens
                if spec.mode == "prefill"
                else spec.generate_tokens
            ),
        ),
    )
    return ConcurrentMemberResult(
        instance_id=spec.instance_id,
        mode=spec.mode,
        status="completed",
        client_ready_ns=900_000_000,
        barrier_release_ns=1_000_000_000,
        first_request_ns=1_000_000_000,
        first_token_ns=(
            1_500_000_000 if spec.mode == "decode" else None
        ),
        last_token_ns=(
            1_500_000_000 if spec.mode == "decode" else None
        ),
        finished_ns=2_000_000_000,
        prompt_tokens=spec.prompt_tokens,
        decode_tokens=spec.generate_tokens,
        native_prompt_tps=(
            native if spec.mode == "prefill" else None
        ),
        native_decode_tps=(
            native if spec.mode == "decode" else None
        ),
        latency_ms=1000.0,
        token_events=events,
        correctness_valid=True,
        raw={"fixture": "deployment-analysis"},
    )


def _finish_phase(
    connection,
    *,
    deployment_id: str,
    deployment_run_id: str,
    phase: str,
    depth: int,
    combined_tps: float,
    retention: float,
) -> str:
    workloads = ConcurrentWorkloadRepository(connection)
    workload = _workload(phase, depth)
    case_id = workloads.put_case(deployment_id, workload)
    run_id = workloads.create_run(
        deployment_run_id=deployment_run_id,
        workload_case_id=case_id,
        phase=phase,
    )
    for ordinal, spec in enumerate(workload.members):
        result = _result(spec, retention=retention)
        standalone = 100.0 if spec.mode == "prefill" else 20.0
        workloads.add_member(
            run_id,
            ordinal=ordinal,
            result=result,
            overlap_prompt_tokens=spec.prompt_tokens,
            overlap_decode_tokens=spec.generate_tokens,
            overlap_prompt_tps=(
                result.native_prompt_tps
                if spec.mode == "prefill"
                else None
            ),
            overlap_decode_tps=(
                result.native_decode_tps
                if spec.mode == "decode"
                else None
            ),
            standalone_baseline_id=None,
            standalone_tps=standalone,
            retention=retention,
            throughput_loss_pct=(1.0 - retention) * 100.0,
            baseline_latency_ms=800.0,
            latency_increase_pct=25.0,
        )
    workloads.finish_run(
        run_id,
        status="completed",
        quality="clean",
        correctness_valid=True,
        barrier_release_ns=1_000_000_000,
        overlap_start_ns=1_000_000_000,
        overlap_end_ns=2_000_000_000,
        overlap_duration_ns=1_000_000_000,
        prompt_tokens=(
            200 if phase == "pp" else 100 if phase in {"pd", "dp"} else 0
        ),
        decode_tokens=(
            40 if phase == "dd" else 20 if phase in {"pd", "dp"} else 0
        ),
        combined_prompt_tps=(
            combined_tps if phase in {"pp", "pd", "dp"} else None
        ),
        combined_decode_tps=(
            combined_tps if phase in {"dd", "pd", "dp"} else None
        ),
        min_retention=retention,
    )
    return run_id


def _seed_subject(
    connection,
    *,
    label: str,
    binary_id: str,
    host_id: str,
    suite_id: str,
    qwen_context: int,
    flash_context: int,
    projected_headroom: int,
    runtime_headroom: int,
    dd_tps: float,
    pp_tps: float,
    retention: float,
    power_samples: tuple[tuple[float, float], ...],
    extra_dd_depth: int | None = None,
) -> tuple[str, str, str, str]:
    candidates = CandidateRepository(connection)
    qwen_id = candidates.put(
        _candidate(f"model:qwen:{label}", qwen_context)
    )
    flash_id = candidates.put(
        _candidate(f"model:flash:{label}", flash_context)
    )
    deployment = DeploymentCandidate(
        instances=(
            ModelInstanceCandidate(
                instance_id="qwen",
                candidate_id=qwen_id,
                role="primary",
                model_artifact_id=f"artifact:qwen:{label}",
                binary_id=binary_id,
                server_identity=f"qwen-{label}",
            ),
            ModelInstanceCandidate(
                instance_id="flash",
                candidate_id=flash_id,
                role="secondary",
                model_artifact_id=f"artifact:flash:{label}",
                binary_id=binary_id,
                server_identity=f"flash-{label}",
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
    qwen_place = placements.put_resolved(
        placement_hash=(label + "q" * 64)[:64],
        candidate_id=qwen_id,
        host_id=host_id,
        binary_id=binary_id,
        fit_attempt_id=None,
        placement=ResolvedPlacement(
            production_context_size=qwen_context,
            n_gpu_layers=40,
            devices=("CUDA0", "Vulkan0"),
            tensor_split=(0.7, 0.3),
        ),
        request={},
        argv=(),
        stdout="",
        stderr="",
        exit_code=0,
    )
    flash_place = placements.put_resolved(
        placement_hash=(label + "f" * 64)[:64],
        candidate_id=flash_id,
        host_id=host_id,
        binary_id=binary_id,
        fit_attempt_id=None,
        placement=ResolvedPlacement(
            production_context_size=flash_context,
            n_gpu_layers=28,
            devices=("CUDA0", "Vulkan0"),
            tensor_split=(0.3, 0.7),
        ),
        request={},
        argv=(),
        stdout="",
        stderr="",
        exit_code=0,
    )

    total = 10_000
    projected = total - projected_headroom - 500
    per_instance = projected // 2
    memory = []
    for instance_id in ("qwen", "flash"):
        for device_id in ("gpu0", "gpu1"):
            model = per_instance // 2
            context = per_instance // 3
            compute = per_instance - model - context
            memory.append(
                PlacementDeviceMemory(
                    instance_id=instance_id,
                    device_id=device_id,
                    model_bytes=model,
                    context_bytes=context,
                    compute_bytes=compute,
                    total_bytes=per_instance,
                    device_total_bytes=total,
                    device_free_bytes=projected_headroom,
                    source="fixture",
                )
            )
    deployment_placement = DeploymentPlacement(
        deployment_candidate_id=deployment_id,
        host_id=host_id,
        instance_placements=(
            DeploymentInstancePlacement(
                instance_id="qwen",
                resolved_placement_id=qwen_place,
            ),
            DeploymentInstancePlacement(
                instance_id="flash",
                resolved_placement_id=flash_place,
            ),
        ),
        device_memory=tuple(memory),
        device_allocations=tuple(
            DeploymentDeviceAllocation(
                device_id=device_id,
                projected_bytes=projected,
                reserved_margin_bytes=500,
                device_total_bytes=total,
                projected_free_bytes=projected_headroom,
            )
            for device_id in ("gpu0", "gpu1")
        ),
        feasibility="feasible",
    )
    deployment_placement_id = DeploymentPlacementRepository(
        connection
    ).put(deployment_placement)

    runs = DeploymentRunRepository(connection)
    deployment_run_id = runs.create(
        deployment_candidate_id=deployment_id,
        deployment_placement_id=deployment_placement_id,
        status="ready",
    )
    for index, (left_power, right_power) in enumerate(power_samples):
        used0 = total - runtime_headroom + index * 100
        used1 = total - runtime_headroom + index * 50
        runs.add_gpu_sample(
            deployment_run_id,
            timestamp_ns=1_000_000_000 + index,
            gpus=(
                GpuTelemetrySample(
                    device="CUDA0",
                    stable_device_key="gpu0",
                    vram_total_bytes=total,
                    vram_used_bytes=min(total, used0),
                    power_w=left_power,
                ),
                GpuTelemetrySample(
                    device="Vulkan0",
                    stable_device_key="gpu1",
                    vram_total_bytes=total,
                    vram_used_bytes=min(total, used1),
                    power_w=right_power,
                ),
            ),
        )

    dd_run = _finish_phase(
        connection,
        deployment_id=deployment_id,
        deployment_run_id=deployment_run_id,
        phase="dd",
        depth=128,
        combined_tps=dd_tps,
        retention=retention,
    )
    _finish_phase(
        connection,
        deployment_id=deployment_id,
        deployment_run_id=deployment_run_id,
        phase="pp",
        depth=128,
        combined_tps=pp_tps,
        retention=retention,
    )
    if extra_dd_depth is not None:
        _finish_phase(
            connection,
            deployment_id=deployment_id,
            deployment_run_id=deployment_run_id,
            phase="dd",
            depth=extra_dd_depth,
            combined_tps=dd_tps - 2.0,
            retention=retention,
        )
    runs.finish(
        deployment_run_id,
        status="completed",
        duration_ns=2_000_000_000,
    )
    return (
        deployment_id,
        deployment_placement_id,
        deployment_run_id,
        dd_run,
    )


def _seed(tmp_path: Path) -> tuple[Database, dict[str, tuple[str, ...]]]:
    database = Database(tmp_path / "deployment-analysis.db")
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"fixture")
    with database.session() as connection:
        environment = EnvironmentRepository(connection)
        binary_id = environment.put_binary(
            sha256=sha256_file(binary),
            kind="llama-server",
            path=str(binary),
            size_bytes=binary.stat().st_size,
            mtime_ns=binary.stat().st_mtime_ns,
            capabilities={},
        )
        host_id = environment.put_host(
            hostname="analysis-host",
            hardware_fingerprint="analysis-host-fingerprint",
            cpu={},
            ram_bytes=1,
            gpus=[],
            os_info={},
        )
        connection.executemany(
            """
            INSERT INTO accelerator_device(
                id, host_id, binary_id, logical_device_name, backend,
                mapping_status, physical_device_key
            )
            VALUES (?, ?, ?, ?, ?, 'mapped', ?)
            """,
            (
                (
                    "accel_cuda0",
                    host_id,
                    binary_id,
                    "CUDA0",
                    "cuda",
                    "gpu0",
                ),
                (
                    "accel_vulkan0",
                    host_id,
                    binary_id,
                    "Vulkan0",
                    "vulkan",
                    "gpu1",
                ),
            ),
        )
        suite_id = WorkloadSuiteRepository(connection).put(
            WorkloadSuite(
                id="deployment-analysis-suite",
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
        subjects = {
            "a": _seed_subject(
                connection,
                label="a",
                binary_id=binary_id,
                host_id=host_id,
                suite_id=suite_id,
                qwen_context=6000,
                flash_context=4000,
                projected_headroom=2200,
                runtime_headroom=2100,
                dd_tps=20.0,
                pp_tps=100.0,
                retention=0.80,
                power_samples=((55.0, 65.0), (65.0, 75.0)),
                extra_dd_depth=256,
            ),
            "b": _seed_subject(
                connection,
                label="b",
                binary_id=binary_id,
                host_id=host_id,
                suite_id=suite_id,
                qwen_context=7000,
                flash_context=5000,
                projected_headroom=1700,
                runtime_headroom=1600,
                dd_tps=25.0,
                pp_tps=90.0,
                retention=0.75,
                power_samples=((70.0, 80.0), (75.0, 85.0)),
            ),
            "c": _seed_subject(
                connection,
                label="c",
                binary_id=binary_id,
                host_id=host_id,
                suite_id=suite_id,
                qwen_context=8000,
                flash_context=6000,
                projected_headroom=1200,
                runtime_headroom=1100,
                dd_tps=18.0,
                pp_tps=110.0,
                retention=0.90,
                power_samples=((50.0, 60.0), (55.0, 65.0)),
            ),
        }

        failed = _seed_subject(
            connection,
            label="d",
            binary_id=binary_id,
            host_id=host_id,
            suite_id=suite_id,
            qwen_context=5000,
            flash_context=4000,
            projected_headroom=2000,
            runtime_headroom=1900,
            dd_tps=15.0,
            pp_tps=80.0,
            retention=0.70,
            power_samples=((50.0, 50.0),),
        )
        failed_run = DeploymentRunRepository(connection).create(
            deployment_candidate_id=failed[0],
            deployment_placement_id=failed[1],
            status="ready",
        )
        DeploymentRunRepository(connection).finish(
            failed_run,
            status="failed",
            duration_ns=1,
            failure_kind="concurrent_workload_failed",
            failure_details={"error": "synthetic failure"},
        )
        connection.execute(
            "DELETE FROM deployment_workload_run WHERE deployment_run_id = ?",
            (failed[2],),
        )
        connection.execute(
            "DELETE FROM deployment_gpu_sample WHERE deployment_run_id = ?",
            (failed[2],),
        )
        subjects["d"] = (failed[0], failed[1], failed_run, "")

        invalid = _seed_subject(
            connection,
            label="e",
            binary_id=binary_id,
            host_id=host_id,
            suite_id=suite_id,
            qwen_context=5500,
            flash_context=4500,
            projected_headroom=1800,
            runtime_headroom=1700,
            dd_tps=17.0,
            pp_tps=85.0,
            retention=0.72,
            power_samples=((52.0, 58.0),),
        )
        connection.execute(
            """
            UPDATE deployment_workload_run
            SET status = 'invalid',
                quality = 'correctness_invalid',
                correctness_valid = 0,
                failure_kind = 'output_validation_failed'
            WHERE deployment_run_id = ?
            """,
            (invalid[2],),
        )
        subjects["e"] = invalid

    return database, subjects


def _filter(path: str, value) -> DeploymentAnalysisFilter:
    return DeploymentAnalysisFilter(path=path, value=value)


def test_memory_matrix_distinguishes_projected_and_runtime(
    tmp_path: Path,
) -> None:
    database, subjects = _seed(tmp_path)
    service = DeploymentAnalysisService(database)
    placement_id = subjects["a"][1]
    run_id = subjects["a"][2]

    matrix = service.memory_matrix(
        placement_id,
        deployment_run_id=run_id,
    )

    assert matrix.devices == ("gpu0", "gpu1")
    rows = {item.key: item for item in matrix.rows}
    assert rows["qwen.model"].source == "projected"
    assert rows["projected_free"].values["gpu0"] == 2200
    assert rows["runtime_peak"].source == "runtime"
    assert rows["runtime_peak"].values["gpu0"] == 8000
    assert rows["runtime_free_min"].values["gpu0"] == 2000


def test_interference_preserves_member_and_latency_metrics(
    tmp_path: Path,
) -> None:
    database, subjects = _seed(tmp_path)
    view = DeploymentAnalysisService(database).interference(
        subjects["a"][3]
    )

    assert view.phase == "dd"
    assert view.combined_decode_tps == pytest.approx(20.0)
    assert view.min_retention == pytest.approx(0.80)
    assert [item.instance_id for item in view.members] == [
        "qwen",
        "flash",
    ]
    assert all(item.retention == pytest.approx(0.80) for item in view.members)
    assert all(
        item.latency_increase_pct == pytest.approx(25.0)
        for item in view.members
    )


def test_exact_hidden_workload_dimension_must_be_filtered(
    tmp_path: Path,
) -> None:
    database, subjects = _seed(tmp_path)
    service = DeploymentAnalysisService(database)
    placement_id = subjects["a"][1]
    common = (
        _filter("deployment.placement_id", placement_id),
        _filter("workload.phase", "dd"),
    )

    with pytest.raises(
        DeploymentAnalysisError,
        match="hidden placement/workload",
    ):
        service.matrix(
            x_path="instance.qwen.candidate.context.size",
            y_path="placement.device.gpu0.projected_free_bytes",
            metric="deployment.combined_tg_tps",
            filters=common,
        )

    matrix = service.matrix(
        x_path="instance.qwen.candidate.context.size",
        y_path="placement.device.gpu0.projected_free_bytes",
        metric="deployment.combined_tg_tps",
        filters=(
            *common,
            _filter("workload.member.qwen.depth_tokens", 128),
            _filter("instance.qwen.resolved.backends.0", "cuda"),
            _filter("instance.qwen.resolved.backends.1", "vulkan"),
        ),
    )
    assert matrix.facets[0].cells[0].value == pytest.approx(20.0)


def test_baseline_comparison_uses_candidate_minus_baseline_sign(
    tmp_path: Path,
) -> None:
    database, subjects = _seed(tmp_path)
    comparison = DeploymentAnalysisService(database).compare(
        subjects["b"][1],
        subjects["a"][1],
        metrics=("deployment.combined_tg_tps",),
        filters=(
            _filter("workload.phase", "dd"),
            _filter("workload.member.qwen.depth_tokens", 128),
        ),
    )

    delta = comparison.deltas[0]
    assert delta.baseline_value == pytest.approx(20.0)
    assert delta.candidate_value == pytest.approx(25.0)
    assert delta.delta == pytest.approx(5.0)
    assert delta.percent_delta == pytest.approx(25.0)


def test_pareto_applies_constraints_before_multiobjective_dominance(
    tmp_path: Path,
) -> None:
    database, subjects = _seed(tmp_path)
    service = DeploymentAnalysisService(database)
    dd_filters = (
        _filter("workload.phase", "dd"),
        _filter("workload.member.qwen.depth_tokens", 128),
    )
    objectives = (
        DeploymentParetoObjective(
            key="dd",
            direction="maximize",
            metric="deployment.combined_tg_tps",
            filters=dd_filters,
        ),
        DeploymentParetoObjective(
            key="pp",
            direction="maximize",
            metric="deployment.combined_pp_tps",
            filters=(_filter("workload.phase", "pp"),),
        ),
        DeploymentParetoObjective(
            key="retention",
            direction="maximize",
            metric="deployment.min_retention",
        ),
        DeploymentParetoObjective(
            key="context",
            direction="maximize",
            metric="deployment.total_validated_context_tokens",
        ),
        DeploymentParetoObjective(
            key="headroom",
            direction="maximize",
            metric="deployment.min_device_headroom_bytes",
        ),
        DeploymentParetoObjective(
            key="power",
            direction="minimize",
            metric="deployment.total_power_avg_w",
        ),
    )

    unconstrained = service.pareto(objectives=objectives)
    assert {
        item.deployment_placement_id
        for item in unconstrained.frontier
    } == {subjects["a"][1], subjects["b"][1], subjects["c"][1]}
    assert subjects["d"][1] in unconstrained.excluded
    assert subjects["e"][1] in unconstrained.excluded

    constrained = service.pareto(
        objectives=objectives,
        constraints=(
            DeploymentMetricConstraint(
                metric="deployment.min_retention",
                operator="ge",
                value=0.80,
            ),
        ),
    )
    assert subjects["b"][1] in constrained.excluded
    assert "constraint failed" in constrained.excluded[subjects["b"][1]]
    points = {
        item.deployment_placement_id: item
        for item in constrained.frontier
    }
    assert points[subjects["a"][1]].values["power"] == pytest.approx(130.0)
    assert points[subjects["c"][1]].values["context"] == pytest.approx(14000.0)


def test_export_round_trip_keeps_failed_rows(tmp_path: Path) -> None:
    database, subjects = _seed(tmp_path)
    service = DeploymentAnalysisService(database)

    json_rows = json.loads(service.export(format_name="json"))
    assert isinstance(json_rows, list)
    assert any(
        row["deployment_run_id"] == subjects["d"][2]
        and row["deployment_status"] == "failed"
        for row in json_rows
    )
    assert any(
        row["deployment_placement_id"] == subjects["e"][1]
        and row["workload_status"] == "invalid"
        and row["correctness_valid"] is False
        for row in json_rows
    )
    assert any(
        row["instance.qwen.retention"] == pytest.approx(0.80)
        for row in json_rows
        if row.get("instance.qwen.retention") is not None
    )

    csv_rows = list(
        csv.DictReader(io.StringIO(service.export(format_name="csv")))
    )
    assert len(csv_rows) == len(json_rows)
    assert any(
        row["deployment_run_id"] == subjects["d"][2]
        and row["deployment_status"] == "failed"
        for row in csv_rows
    )
