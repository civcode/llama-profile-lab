"""Joint deployment planner capability and memory-feasibility tests."""

from __future__ import annotations

from pathlib import Path

from llama_profile_lab.db import (
    AcceleratorDeviceRepository,
    CandidateRepository,
    Database,
    DeploymentCandidateRepository,
    DeploymentPlanRepository,
    EnvironmentRepository,
    WorkloadSuiteRepository,
)
from llama_profile_lab.domain import (
    AbsoluteDepth,
    AcceleratorDevice,
    BackendPair,
    Candidate,
    ComputeConfig,
    ContextConfig,
    DeploymentCandidate,
    DeploymentPlacementRequest,
    DeploymentSearchDimension,
    DeploymentSearchSpace,
    DeploymentWorkloadMix,
    FitConfig,
    HostResourcePolicy,
    ModelInstanceCandidate,
    ModelSelection,
    PlacementConfig,
    WorkloadSuite,
)
from llama_profile_lab.domain.device import (
    MemoryEstimateDevice,
    MemoryEstimateOutput,
    MemoryEstimateResolved,
)
from llama_profile_lab.domain.workload import PrefillSuiteCase
from llama_profile_lab.execution.host import BasicHostInfo
from llama_profile_lab.execution.memory_estimator import MemoryEstimateObservation
from llama_profile_lab.planning import (
    DeploymentEstimatorInput,
    DeploymentPlannerService,
)


HOST = BasicHostInfo(
    hostname="test-host",
    hardware_fingerprint="test-fingerprint",
    cpu={},
    ram_bytes=64 * 1024 * 1024 * 1024,
    gpus=[],
    os_info={},
)


class FakeEstimator:
    def __init__(self, database: Database, size_fn) -> None:
        self.database = database
        self.size_fn = size_fn
        self.calls = 0

    def estimate(
        self,
        candidate_id: str,
        *,
        helper_binary_id: str,
        model_path: Path,
        selected_devices: tuple[str, ...] | None = None,
        placement_constraints=None,
        timeout_seconds: float | None = 300.0,
    ) -> MemoryEstimateObservation:
        self.calls += 1
        assert selected_devices is not None and selected_devices
        with self.database.session() as connection:
            candidate = CandidateRepository(connection).get(candidate_id)
        assert candidate is not None
        total = self.size_fn(candidate)
        weights = (
            placement_constraints.tensor_split
            if placement_constraints is not None
            and placement_constraints.tensor_split is not None
            else tuple(1.0 for _ in selected_devices)
        )
        weight_total = sum(weights)
        allocated: list[int] = []
        remaining = total
        for index, weight in enumerate(weights):
            if index == len(weights) - 1:
                value = remaining
            else:
                value = int(total * weight / weight_total)
                remaining -= value
            allocated.append(value)
        rows = tuple(
            MemoryEstimateDevice(
                logical_device_name=device,
                model_bytes=value,
                context_bytes=0,
                compute_bytes=0,
                total_bytes=value,
                device_total_bytes=1000,
                device_free_bytes=1000,
            )
            for device, value in zip(selected_devices, allocated, strict=True)
        )
        output = MemoryEstimateOutput(
            devices=rows,
            resolved=MemoryEstimateResolved(
                n_gpu_layers=48,
                devices=selected_devices,
                split_mode=(
                    placement_constraints.split_mode
                    if placement_constraints is not None
                    else "layer"
                ),
                main_gpu=(
                    placement_constraints.main_gpu
                    if placement_constraints is not None
                    else 0
                ),
                tensor_split=(
                    placement_constraints.tensor_split
                    if placement_constraints is not None
                    else None
                ),
                override_tensor=(),
            ),
        )
        return MemoryEstimateObservation(
            estimate_id=f"memest_{candidate_id}_{self.calls}",
            attempt_id=f"attempt_{self.calls}",
            output=output,
            cache_hit=False,
        )


def make_candidate(
    model_id: str,
    *,
    context: int = 8192,
    kv: str = "f16",
) -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id=model_id),
        context=ContextConfig(
            size=context,
            cache_type_k=kv,
            cache_type_v=kv,
        ),
        compute=ComputeConfig(
            flash_attn="on",
            batch_size=2048,
            ubatch_size=512,
        ),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=0, min_context=4096),
        ),
    )


def seed(
    tmp_path: Path,
    *,
    qwen_request: DeploymentPlacementRequest | None = None,
    flash_request: DeploymentPlacementRequest | None = None,
    policy: HostResourcePolicy | None = None,
) -> tuple[Database, str, str]:
    database = Database(tmp_path / "planner.db")
    qwen = make_candidate("model:qwen")
    flash = make_candidate("model:flash")
    with database.session() as connection:
        candidate_repo = CandidateRepository(connection)
        qwen_id = candidate_repo.put(qwen)
        flash_id = candidate_repo.put(flash)
        suite_id = WorkloadSuiteRepository(connection).put(
            WorkloadSuite(
                id="suite_1",
                cases=(
                    PrefillSuiteCase(
                        prompt_tokens=128,
                        depth=AbsoluteDepth(tokens=128),
                    ),
                ),
            )
        )
        environment = EnvironmentRepository(connection)
        server_id = environment.put_binary(
            sha256="1" * 64,
            kind="llama-server",
            path="/tmp/llama-server",
            size_bytes=1,
            mtime_ns=1,
            capabilities={
                "kind": "llama-server",
                "options": [
                    "--device",
                    "--tensor-split",
                    "--split-mode",
                ],
            },
        )
        helper_id = environment.put_binary(
            sha256="2" * 64,
            kind="llama-memory-estimator",
            path="/tmp/llama-memory-estimator",
            size_bytes=1,
            mtime_ns=1,
            capabilities={
                "kind": "llama-memory-estimator",
                "options": [
                    "--device",
                    "--tensor-split",
                    "--split-mode",
                ],
            },
        )
        host_id = environment.put_host(
            hostname=HOST.hostname,
            hardware_fingerprint=HOST.hardware_fingerprint,
            cpu=HOST.cpu,
            ram_bytes=HOST.ram_bytes,
            gpus=HOST.gpus,
            os_info=HOST.os_info,
        )
        devices = (
            AcceleratorDevice(
                logical_device_name="GPU0",
                backend="CUDA",
                mapping_status="mapped",
                physical_device_key="pci:0000:01:00.0",
                pci_bus_id="0000:01:00.0",
                total_memory_bytes=1000,
                free_memory_bytes=1000,
            ),
            AcceleratorDevice(
                logical_device_name="GPU1",
                backend="VULKAN",
                mapping_status="mapped",
                physical_device_key="pci:0000:02:00.0",
                pci_bus_id="0000:02:00.0",
                total_memory_bytes=1000,
                free_memory_bytes=1000,
            ),
        )
        AcceleratorDeviceRepository(connection).put_inventory(
            host_id=host_id,
            binary_id=server_id,
            devices=devices,
            raw_output="fixture",
        )
        AcceleratorDeviceRepository(connection).put_inventory(
            host_id=host_id,
            binary_id=helper_id,
            devices=devices,
            raw_output="fixture",
        )
        deployment = DeploymentCandidate(
            instances=(
                ModelInstanceCandidate(
                    instance_id="qwen",
                    candidate_id=qwen_id,
                    role="primary",
                    model_artifact_id="model:qwen",
                    binary_id=server_id,
                    requested_placement=qwen_request
                    or DeploymentPlacementRequest(devices=("GPU0",)),
                    server_identity="qwen",
                ),
                ModelInstanceCandidate(
                    instance_id="flash",
                    candidate_id=flash_id,
                    role="secondary",
                    model_artifact_id="model:flash",
                    binary_id=server_id,
                    requested_placement=flash_request
                    or DeploymentPlacementRequest(devices=("GPU1",)),
                    server_identity="flash",
                ),
            ),
            resource_policy=policy or HostResourcePolicy(),
            workload_mix=DeploymentWorkloadMix(
                workload_suite_id=suite_id
            ),
        )
        deployment_id = DeploymentCandidateRepository(connection).put(
            deployment
        )
    return database, deployment_id, helper_id


def inputs(helper_id: str) -> tuple[DeploymentEstimatorInput, ...]:
    return (
        DeploymentEstimatorInput(
            "qwen",
            helper_id,
            Path("/tmp/qwen.gguf"),
        ),
        DeploymentEstimatorInput(
            "flash",
            helper_id,
            Path("/tmp/flash.gguf"),
        ),
    )


def test_planner_counts_capability_memory_and_valid_cases(
    tmp_path: Path,
) -> None:
    database, deployment_id, helper_id = seed(tmp_path)

    def size(candidate: Candidate) -> int:
        if candidate.model.target_model_id == "model:qwen":
            return 400
        return 1200 if candidate.context.size > 8192 else 300

    estimator = FakeEstimator(database, size)
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.requested_placement.devices",
                values=(("GPU0",), ("MISSING",)),
            ),
            DeploymentSearchDimension(
                path="instances.flash.context.size",
                values=(8192, 32768),
            ),
        )
    )
    summary = DeploymentPlannerService(
        database,
        memory_estimator=estimator,
        host_detector=lambda: HOST,
    ).plan(deployment_id, search, inputs(helper_id))

    assert summary.raw_combinations == 4
    assert summary.capability_rejected == 2
    assert summary.memory_rejected == 1
    assert summary.valid_count == 1
    assert summary.plan_id is not None
    with database.session() as connection:
        cases = DeploymentPlanRepository(connection).cases(summary.plan_id)
        rejections = connection.execute(
            """
            SELECT reason
            FROM deployment_rejection
            ORDER BY reason
            """
        ).fetchall()
    assert len(cases) == 1
    assert [row["reason"] for row in rejections] == [
        "device_memory_exceeded",
        "unsupported_device",
        "unsupported_device",
    ]


def test_exact_fit_and_gpu_specific_overflow(tmp_path: Path) -> None:
    database, deployment_id, helper_id = seed(
        tmp_path,
        qwen_request=DeploymentPlacementRequest(devices=("GPU0",)),
        flash_request=DeploymentPlacementRequest(devices=("GPU0",)),
    )
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="resource_policy.device_memory_margin_bytes.GPU0",
                values=(0,),
            ),
        )
    )
    exact = FakeEstimator(
        database,
        lambda candidate: (
            700
            if candidate.model.target_model_id == "model:qwen"
            else 300
        ),
    )
    summary = DeploymentPlannerService(
        database,
        memory_estimator=exact,
        host_detector=lambda: HOST,
    ).preview(deployment_id, search, inputs(helper_id))
    assert summary.valid_count == 1
    assert summary.memory_rejected == 0

    overflow = FakeEstimator(
        database,
        lambda candidate: (
            800
            if candidate.model.target_model_id == "model:qwen"
            else 300
        ),
    )
    rejected = DeploymentPlannerService(
        database,
        memory_estimator=overflow,
        host_detector=lambda: HOST,
    ).preview(deployment_id, search, inputs(helper_id))
    assert rejected.valid_count == 0
    assert rejected.memory_rejected == 1


def test_gpu1_only_overflow_and_backend_pair_rejection(
    tmp_path: Path,
) -> None:
    database, deployment_id, helper_id = seed(
        tmp_path,
        qwen_request=DeploymentPlacementRequest(devices=("GPU1",)),
        flash_request=DeploymentPlacementRequest(devices=("GPU1",)),
    )
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.context.size",
                values=(8192,),
            ),
        )
    )
    estimator = FakeEstimator(
        database,
        lambda candidate: (
            800
            if candidate.model.target_model_id == "model:qwen"
            else 300
        ),
    )
    summary = DeploymentPlannerService(
        database,
        memory_estimator=estimator,
        host_detector=lambda: HOST,
    ).preview(deployment_id, search, inputs(helper_id))
    assert summary.memory_rejected == 1

    database2, deployment2, helper2 = seed(
        tmp_path / "pair",
        qwen_request=DeploymentPlacementRequest(devices=("GPU0",)),
        flash_request=DeploymentPlacementRequest(devices=("GPU1",)),
        policy=HostResourcePolicy(
            allowed_backend_pairs=(
                BackendPair(left="CUDA", right="CUDA"),
            )
        ),
    )
    summary2 = DeploymentPlannerService(
        database2,
        memory_estimator=FakeEstimator(
            database2,
            lambda candidate: 200,
        ),
        host_detector=lambda: HOST,
    ).preview(deployment2, search, inputs(helper2))
    assert summary2.capability_rejected == 1
    assert summary2.valid_count == 0


def test_both_models_can_split_across_both_gpus(
    tmp_path: Path,
) -> None:
    request = DeploymentPlacementRequest(
        devices=("GPU0", "GPU1"),
        tensor_split=(1.0, 1.0),
    )
    database, deployment_id, helper_id = seed(
        tmp_path,
        qwen_request=request,
        flash_request=request,
    )
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="resource_policy.device_memory_margin_bytes.GPU0",
                values=(50,),
            ),
        )
    )

    summary = DeploymentPlannerService(
        database,
        memory_estimator=FakeEstimator(
            database,
            lambda candidate: 400,
        ),
        host_detector=lambda: HOST,
    ).preview(deployment_id, search, inputs(helper_id))

    assert summary.valid_count == 1
    assert summary.memory_rejected == 0


class FailingEstimator:
    def estimate(self, *args, **kwargs):
        raise ValueError("synthetic estimate failure")


def test_memory_estimate_failure_is_counted_and_persisted(
    tmp_path: Path,
) -> None:
    database, deployment_id, helper_id = seed(tmp_path)
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.context.size",
                values=(8192,),
            ),
        )
    )

    summary = DeploymentPlannerService(
        database,
        memory_estimator=FailingEstimator(),
        host_detector=lambda: HOST,
    ).plan(deployment_id, search, inputs(helper_id))

    assert summary.estimate_failed == 1
    assert summary.valid_count == 0
    with database.session() as connection:
        reasons = [
            row["reason"]
            for row in connection.execute(
                "SELECT reason FROM deployment_rejection"
            ).fetchall()
        ]
    assert reasons == ["memory_estimate_failed"]


def test_invalid_split_and_kv_precision_rescue(tmp_path: Path) -> None:
    database, deployment_id, helper_id = seed(tmp_path)
    invalid = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.requested_placement.split_mode",
                values=("bogus",),
            ),
        )
    )
    service = DeploymentPlannerService(
        database,
        memory_estimator=FakeEstimator(
            database,
            lambda candidate: 200,
        ),
        host_detector=lambda: HOST,
    )
    invalid_summary = service.preview(
        deployment_id,
        invalid,
        inputs(helper_id),
    )
    assert invalid_summary.capability_rejected == 1

    rescue = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.context.size",
                values=(32768,),
            ),
            DeploymentSearchDimension(
                path="instances.qwen.context.cache_type_k",
                values=("f16", "q8_0"),
            ),
            DeploymentSearchDimension(
                path="instances.qwen.context.cache_type_v",
                values=("f16", "q8_0"),
            ),
        ),
        constraints=(
            "instances.qwen.context.cache_type_k == "
            "instances.qwen.context.cache_type_v",
        ),
    )
    estimator = FakeEstimator(
        database,
        lambda candidate: (
            1200
            if candidate.model.target_model_id == "model:qwen"
            and candidate.context.cache_type_k == "f16"
            else 600
            if candidate.model.target_model_id == "model:qwen"
            else 300
        ),
    )
    rescued = DeploymentPlannerService(
        database,
        memory_estimator=estimator,
        host_detector=lambda: HOST,
    ).preview(deployment_id, rescue, inputs(helper_id))
    assert rescued.raw_combinations == 4
    assert rescued.rejected_by_constraints == 2
    assert rescued.memory_rejected == 1
    assert rescued.valid_count == 1


def test_high_context_rejected_for_single_and_split_options(
    tmp_path: Path,
) -> None:
    database, deployment_id, helper_id = seed(tmp_path)
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.context.size",
                values=(65536,),
            ),
            DeploymentSearchDimension(
                path="instances.qwen.requested_placement.devices",
                values=(("GPU0",), ("GPU0", "GPU1")),
            ),
        )
    )
    estimator = FakeEstimator(
        database,
        lambda candidate: (
            1800
            if candidate.model.target_model_id == "model:qwen"
            else 300
        ),
    )
    summary = DeploymentPlannerService(
        database,
        memory_estimator=estimator,
        host_detector=lambda: HOST,
    ).preview(deployment_id, search, inputs(helper_id))

    assert summary.raw_combinations == 2
    assert summary.memory_rejected == 2
    assert summary.valid_count == 0
