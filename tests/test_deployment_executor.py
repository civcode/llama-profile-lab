"""Integration coverage for simultaneous deployment server execution."""

from __future__ import annotations

import stat
import threading
import time
from pathlib import Path

import pytest

from llama_profile_lab.db import (
    CandidateRepository,
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
    AcceleratorDevice,
    Candidate,
    ComputeConfig,
    ContextConfig,
    DeploymentCandidate,
    DeploymentDeviceAllocation,
    DeploymentInstancePlacement,
    DeploymentPlacement,
    DeploymentPlacementRequest,
    DeploymentWorkloadMix,
    FitConfig,
    GpuTelemetrySample,
    HostResourcePolicy,
    ModelInstanceCandidate,
    ModelSelection,
    PlacementConfig,
    PrefillSuiteCase,
    ResolvedPlacement,
    WorkloadSuite,
)
from llama_profile_lab.execution import (
    BasicHostInfo,
    DeploymentExecutionError,
    DeploymentExecutor,
    DeviceInventoryResult,
    DeploymentServerInput,
    HostLock,
    ManagedServerProcess,
)
from llama_profile_lab.llama import sha256_file


HOST = BasicHostInfo(
    hostname="deployment-executor-test",
    hardware_fingerprint="deployment-executor-fingerprint",
    cpu={},
    ram_bytes=64 * 1024**3,
    gpus=[],
    os_info={},
)


class StaticGpuProvider:
    def __init__(self, *samples: GpuTelemetrySample) -> None:
        self.samples = tuple(samples)

    def sample(self) -> tuple[GpuTelemetrySample, ...]:
        return self.samples


class StaticDeviceInventory:
    def __init__(self, *devices: AcceleratorDevice) -> None:
        self.devices = tuple(devices)

    def inspect(
        self,
        binary_id: str,
        *,
        timeout_seconds: float | None = 30.0,
        cancel_event=None,
    ) -> DeviceInventoryResult:
        del timeout_seconds, cancel_event
        return DeviceInventoryResult(
            host_id="executor-host",
            binary_id=binary_id,
            devices=self.devices,
            stdout="",
            stderr="",
        )


def _inventory(
    *,
    include_vulkan: bool = True,
    vulkan_key: str = "pci:0000:02:00.0",
) -> StaticDeviceInventory:
    devices = [
        AcceleratorDevice(
            logical_device_name="CUDA0",
            backend="cuda",
            mapping_status="mapped",
            physical_device_key="pci:0000:01:00.0",
            pci_bus_id="0000:01:00.0",
        )
    ]
    if include_vulkan:
        devices.append(
            AcceleratorDevice(
                logical_device_name="Vulkan0",
                backend="vulkan",
                mapping_status="mapped",
                physical_device_key=vulkan_key,
                pci_bus_id=(
                    "0000:02:00.0"
                    if vulkan_key == "pci:0000:02:00.0"
                    else None
                ),
            )
        )
    return StaticDeviceInventory(*devices)


def _candidate(model_id: str) -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id=model_id),
        context=ContextConfig(
            size=8192,
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
            fit=FitConfig(target_mib=0, min_context=4096),
        ),
    )


def _write_fake_server(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import http.server
import os
import signal
import subprocess
import sys
import time

args = sys.argv[1:]

def value(name):
    return args[args.index(name) + 1]

host = value("--host")
port = int(value("--port"))
model = value("--model")

if "fail" in os.path.basename(model):
    print("synthetic startup failure", file=sys.stderr, flush=True)
    raise SystemExit(7)
if "oom" in os.path.basename(model):
    print("out of memory", file=sys.stderr, flush=True)
    raise SystemExit(9)

child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"]
)
with open(model + ".childpid", "w", encoding="utf-8") as handle:
    handle.write(str(child.pid))

if "crashchild" in os.path.basename(model):
    print("synthetic parent crash", file=sys.stderr, flush=True)
    raise SystemExit(11)

if "slow" in os.path.basename(model):
    time.sleep(0.35)
if "neverready" in os.path.basename(model):
    time.sleep(60)

stop = False

def request_stop(signum, frame):
    global stop
    stop = True

signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return

server = http.server.HTTPServer((host, port), Handler)
server.timeout = 0.05
print(f"ready:{port}", flush=True)
while not stop:
    server.handle_request()
server.server_close()
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _server_capabilities() -> dict[str, object]:
    return {
        "kind": "llama-server",
        "options": [
            "--model",
            "--host",
            "--port",
            "--ctx-size",
            "--batch-size",
            "--ubatch-size",
            "--cache-type-k",
            "--cache-type-v",
            "--parallel",
            "--flash-attn",
            "--n-gpu-layers",
            "--device",
        ],
    }


def _seed(
    tmp_path: Path,
    *,
    qwen_name: str = "qwen.gguf",
    flash_name: str = "flash.gguf",
    margin_bytes: int = 100,
) -> tuple[Database, str, tuple[DeploymentServerInput, ...], Path, Path]:
    database = Database(tmp_path / "executor.db")
    server_path = tmp_path / "llama-server"
    _write_fake_server(server_path)

    qwen_path = tmp_path / qwen_name
    flash_path = tmp_path / flash_name
    qwen_path.write_bytes(b"qwen")
    flash_path.write_bytes(b"flash")

    with database.session() as connection:
        candidates = CandidateRepository(connection)
        qwen_id = candidates.put(_candidate("model:qwen"))
        flash_id = candidates.put(_candidate("model:flash"))
        suite_id = WorkloadSuiteRepository(connection).put(
            WorkloadSuite(
                id="executor-suite",
                cases=(
                    PrefillSuiteCase(
                        prompt_tokens=64,
                        depth=AbsoluteDepth(tokens=64),
                    ),
                ),
            )
        )
        environment = EnvironmentRepository(connection)
        binary_id = environment.put_binary(
            sha256=sha256_file(server_path),
            kind="llama-server",
            path=str(server_path),
            size_bytes=server_path.stat().st_size,
            mtime_ns=server_path.stat().st_mtime_ns,
            capabilities=_server_capabilities(),
        )
        host_id = environment.put_host(
            hostname=HOST.hostname,
            hardware_fingerprint=HOST.hardware_fingerprint,
            cpu=HOST.cpu,
            ram_bytes=HOST.ram_bytes,
            gpus=HOST.gpus,
            os_info=HOST.os_info,
        )

        deployment = DeploymentCandidate(
            instances=(
                ModelInstanceCandidate(
                    instance_id="qwen",
                    candidate_id=qwen_id,
                    role="primary",
                    model_artifact_id="model:qwen",
                    binary_id=binary_id,
                    requested_placement=DeploymentPlacementRequest(
                        devices=("CUDA0",)
                    ),
                    server_identity="qwen",
                ),
                ModelInstanceCandidate(
                    instance_id="flash",
                    candidate_id=flash_id,
                    role="secondary",
                    model_artifact_id="model:flash",
                    binary_id=binary_id,
                    requested_placement=DeploymentPlacementRequest(
                        devices=("Vulkan0",)
                    ),
                    server_identity="flash",
                ),
            ),
            resource_policy=HostResourcePolicy(
                device_memory_margin_bytes={
                    "pci:0000:01:00.0": margin_bytes,
                    "pci:0000:02:00.0": margin_bytes,
                }
            ),
            workload_mix=DeploymentWorkloadMix(
                workload_suite_id=suite_id
            ),
        )
        deployment_id = DeploymentCandidateRepository(connection).put(
            deployment
        )

        placements = PlacementRepository(connection)
        qwen_placement = placements.put_resolved(
            placement_hash="1" * 64,
            candidate_id=qwen_id,
            host_id=host_id,
            binary_id=binary_id,
            fit_attempt_id=None,
            placement=ResolvedPlacement(
                production_context_size=8192,
                n_gpu_layers=32,
                devices=("CUDA0",),
            ),
            request={},
            argv=(),
            stdout="",
            stderr="",
            exit_code=0,
        )
        flash_placement = placements.put_resolved(
            placement_hash="2" * 64,
            candidate_id=flash_id,
            host_id=host_id,
            binary_id=binary_id,
            fit_attempt_id=None,
            placement=ResolvedPlacement(
                production_context_size=8192,
                n_gpu_layers=24,
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
                        instance_id="qwen",
                        resolved_placement_id=qwen_placement,
                    ),
                    DeploymentInstancePlacement(
                        instance_id="flash",
                        resolved_placement_id=flash_placement,
                    ),
                ),
                device_allocations=(
                    DeploymentDeviceAllocation(
                        device_id="pci:0000:01:00.0",
                        projected_bytes=600,
                        reserved_margin_bytes=margin_bytes,
                        device_total_bytes=1000,
                        projected_free_bytes=1000 - 600 - margin_bytes,
                    ),
                    DeploymentDeviceAllocation(
                        device_id="pci:0000:02:00.0",
                        projected_bytes=500,
                        reserved_margin_bytes=margin_bytes,
                        device_total_bytes=1000,
                        projected_free_bytes=1000 - 500 - margin_bytes,
                    ),
                ),
                feasibility="feasible",
            )
        )

    return (
        database,
        deployment_placement_id,
        (
            DeploymentServerInput("qwen", qwen_path),
            DeploymentServerInput("flash", flash_path),
        ),
        qwen_path,
        flash_path,
    )


def _provider(*, free0: int = 200, free1: int = 200) -> StaticGpuProvider:
    return StaticGpuProvider(
        GpuTelemetrySample(
            device="0000:01:00.0",
            stable_device_key="pci:0000:01:00.0",
            vram_total_bytes=1000,
            vram_used_bytes=1000 - free0,
        ),
        GpuTelemetrySample(
            device="0000:02:00.0",
            stable_device_key="pci:0000:02:00.0",
            vram_total_bytes=1000,
            vram_used_bytes=1000 - free1,
        ),
    )


def _executor(
    database: Database,
    *,
    provider: StaticGpuProvider | None = None,
    device_inventory: StaticDeviceInventory | None = None,
) -> DeploymentExecutor:
    return DeploymentExecutor(
        database,
        host_detector=lambda: HOST,
        gpu_provider=provider or _provider(),
        device_inventory=device_inventory or _inventory(),
    )


def _process_terminated(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return True
    parts = stat_path.read_text(encoding="utf-8").split()
    return len(parts) > 2 and parts[2] == "Z"


def _assert_process_terminates(pid: int) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if _process_terminated(pid):
            return
        time.sleep(0.02)
    pytest.fail(f"process {pid} is still alive")


def _child_pid(model_path: Path) -> int:
    return int(
        Path(str(model_path) + ".childpid").read_text(
            encoding="utf-8"
        )
    )


def test_two_servers_reach_ready_and_shutdown_cleanly(tmp_path: Path) -> None:
    database, placement_id, inputs, qwen_path, flash_path = _seed(
        tmp_path,
        flash_name="flash-slow.gguf",
    )

    summary = _executor(database).execute(
        placement_id,
        inputs,
        readiness_timeout_seconds=2.0,
        residency_hold_seconds=0.05,
    )

    assert summary.status == "completed"
    assert len(summary.members) == 2
    assert len({item.endpoint for item in summary.members}) == 2
    assert all(item.ready_at is not None for item in summary.members)
    assert all(item.exit_code == 0 for item in summary.members)

    with database.session() as connection:
        repository = DeploymentRunRepository(connection)
        run = repository.get(summary.run_id)
        members = repository.members(summary.run_id)
        gpu_samples = repository.gpu_samples(summary.run_id)
    assert run is not None and run.status == "completed"
    assert len(gpu_samples) >= 2
    assert all(len(item.gpus) == 2 for item in gpu_samples)
    assert [item.member_status for item in members] == ["stopped", "stopped"]
    assert all(item.endpoint for item in members)
    assert all("ready:" in item.stdout for item in members)

    for member in summary.members:
        assert member.pid is not None
        _assert_process_terminates(member.pid)
    _assert_process_terminates(_child_pid(qwen_path))
    _assert_process_terminates(_child_pid(flash_path))


def test_later_server_failure_tears_down_ready_member(tmp_path: Path) -> None:
    database, placement_id, inputs, qwen_path, _ = _seed(
        tmp_path,
        flash_name="flash-fail.gguf",
    )

    with pytest.raises(DeploymentExecutionError) as captured:
        _executor(database).execute(
            placement_id,
            inputs,
            readiness_timeout_seconds=1.0,
        )

    run_id = captured.value.run_id
    assert run_id is not None
    with database.session() as connection:
        run = DeploymentRunRepository(connection).get(run_id)
        members = DeploymentRunRepository(connection).members(run_id)
    assert run is not None and run.status == "failed"
    assert run.failure_kind == "server_start_failed"
    assert all(item.member_status == "failed" for item in members)
    qwen = next(item for item in members if item.instance_id == "qwen")
    assert qwen.pid is not None
    _assert_process_terminates(qwen.pid)
    _assert_process_terminates(_child_pid(qwen_path))


def test_first_server_failure_tears_down_deployment(
    tmp_path: Path,
) -> None:
    database, placement_id, inputs, _, flash_path = _seed(
        tmp_path,
        qwen_name="qwen-fail.gguf",
    )

    with pytest.raises(DeploymentExecutionError) as captured:
        _executor(database).execute(
            placement_id,
            inputs,
            readiness_timeout_seconds=1.0,
        )

    assert captured.value.failure_kind == "server_start_failed"
    with database.session() as connection:
        members = DeploymentRunRepository(connection).members(
            captured.value.run_id or ""
        )
    assert all(item.member_status == "failed" for item in members)
    flash = next(item for item in members if item.instance_id == "flash")
    if flash.pid is not None:
        _assert_process_terminates(flash.pid)
    child_file = Path(str(flash_path) + ".childpid")
    if child_file.exists():
        _assert_process_terminates(_child_pid(flash_path))


def test_binary_drift_fails_before_server_start(tmp_path: Path) -> None:
    database, placement_id, inputs, _, _ = _seed(tmp_path)
    with database.session() as connection:
        placement = DeploymentPlacementRepository(connection).get(placement_id)
        assert placement is not None
        deployment = DeploymentCandidateRepository(connection).get(
            placement.deployment_candidate_id
        )
        assert deployment is not None
        binary_id = deployment.instances[0].binary_id
        binary = EnvironmentRepository(connection).get_binary(binary_id)
        assert binary is not None
        binary_path = Path(binary.path)

    with binary_path.open("a", encoding="utf-8") as handle:
        handle.write("\n# drift\n")

    with pytest.raises(DeploymentExecutionError, match="changed on disk") as captured:
        _executor(database).execute(placement_id, inputs)

    assert captured.value.run_id is not None
    with database.session() as connection:
        run = DeploymentRunRepository(connection).get(captured.value.run_id)
        members = DeploymentRunRepository(connection).members(
            captured.value.run_id
        )
    assert run is not None and run.status == "failed"
    assert members == ()


def test_missing_model_fails_before_server_start(tmp_path: Path) -> None:
    database, placement_id, inputs, qwen_path, _ = _seed(tmp_path)
    qwen_path.unlink()

    with pytest.raises(
        DeploymentExecutionError,
        match="model path does not exist",
    ) as captured:
        _executor(database).execute(placement_id, inputs)

    assert captured.value.run_id is not None
    with database.session() as connection:
        run = DeploymentRunRepository(connection).get(captured.value.run_id)
    assert run is not None and run.status == "failed"


def test_parent_crash_still_cleans_process_group(tmp_path: Path) -> None:
    database, placement_id, inputs, qwen_path, _ = _seed(
        tmp_path,
        qwen_name="qwen-crashchild.gguf",
    )

    with pytest.raises(DeploymentExecutionError) as captured:
        _executor(database).execute(
            placement_id,
            inputs,
            readiness_timeout_seconds=1.0,
        )

    assert captured.value.failure_kind == "server_start_failed"
    _assert_process_terminates(_child_pid(qwen_path))


def test_runtime_oom_is_classified_from_member_logs(tmp_path: Path) -> None:
    database, placement_id, inputs, _, _ = _seed(
        tmp_path,
        qwen_name="qwen-oom.gguf",
    )

    with pytest.raises(DeploymentExecutionError) as captured:
        _executor(database).execute(
            placement_id,
            inputs,
            readiness_timeout_seconds=1.0,
        )

    assert captured.value.failure_kind == "server_oom"
    with database.session() as connection:
        run = DeploymentRunRepository(connection).get(
            captured.value.run_id or ""
        )
    assert run is not None and run.failure_kind == "server_oom"


class CleanupFailingServer(ManagedServerProcess):
    def close(self) -> None:
        super().close()
        raise RuntimeError("synthetic cleanup failure")


def test_cleanup_failure_is_persisted_after_process_stop(
    tmp_path: Path,
) -> None:
    database, placement_id, inputs, _, _ = _seed(tmp_path)
    executor = DeploymentExecutor(
        database,
        host_detector=lambda: HOST,
        gpu_provider=_provider(),
        device_inventory=_inventory(),
        server_process_factory=CleanupFailingServer,
    )

    with pytest.raises(DeploymentExecutionError) as captured:
        executor.execute(
            placement_id,
            inputs,
            readiness_timeout_seconds=2.0,
        )

    assert captured.value.failure_kind == "member_crash"
    with database.session() as connection:
        run = DeploymentRunRepository(connection).get(
            captured.value.run_id or ""
        )
        members = DeploymentRunRepository(connection).members(
            captured.value.run_id or ""
        )
    assert run is not None and run.status == "failed"
    assert all(
        "synthetic cleanup failure" in (item.cleanup_error or "")
        for item in members
    )
    for item in members:
        if item.pid is not None:
            _assert_process_terminates(item.pid)


def test_readiness_timeout_tears_down_all_started_members(
    tmp_path: Path,
) -> None:
    database, placement_id, inputs, qwen_path, flash_path = _seed(
        tmp_path,
        flash_name="flash-neverready.gguf",
    )

    with pytest.raises(DeploymentExecutionError) as captured:
        _executor(database).execute(
            placement_id,
            inputs,
            readiness_timeout_seconds=0.2,
        )

    assert captured.value.failure_kind == "member_timeout"
    with database.session() as connection:
        members = DeploymentRunRepository(connection).members(
            captured.value.run_id or ""
        )
    for member in members:
        if member.pid is not None:
            _assert_process_terminates(member.pid)
    _assert_process_terminates(_child_pid(qwen_path))
    _assert_process_terminates(_child_pid(flash_path))


def test_cancellation_during_residency_tears_down_both(
    tmp_path: Path,
) -> None:
    database, placement_id, inputs, _, _ = _seed(tmp_path)
    cancel = threading.Event()
    observed: list[DeploymentExecutionError] = []

    def run() -> None:
        try:
            _executor(database).execute(
                placement_id,
                inputs,
                readiness_timeout_seconds=2.0,
                residency_hold_seconds=5.0,
                cancel_event=cancel,
            )
        except DeploymentExecutionError as exc:
            observed.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.25)
    cancel.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert len(observed) == 1
    assert observed[0].status == "cancelled"
    with database.session() as connection:
        run_record = DeploymentRunRepository(connection).get(
            observed[0].run_id or ""
        )
        members = DeploymentRunRepository(connection).members(
            observed[0].run_id or ""
        )
    assert run_record is not None and run_record.status == "cancelled"
    assert all(item.member_status == "cancelled" for item in members)
    for member in members:
        if member.pid is not None:
            _assert_process_terminates(member.pid)


def test_host_lock_excludes_deployment_execution(tmp_path: Path) -> None:
    database, placement_id, inputs, _, _ = _seed(tmp_path)
    lock_path = Path(database.path).with_suffix(
        Path(database.path).suffix + ".host.lock"
    )

    with HostLock(lock_path):
        with pytest.raises(DeploymentExecutionError, match="already held"):
            _executor(database).execute(placement_id, inputs)


def test_device_inventory_change_fails_before_server_start(
    tmp_path: Path,
) -> None:
    database, placement_id, inputs, qwen_path, flash_path = _seed(tmp_path)

    with pytest.raises(DeploymentExecutionError) as captured:
        _executor(
            database,
            device_inventory=_inventory(include_vulkan=False),
        ).execute(
            placement_id,
            inputs,
            readiness_timeout_seconds=1.0,
        )

    assert captured.value.failure_kind == "device_capability_mismatch"
    assert "Vulkan0" in str(captured.value)
    assert not Path(str(qwen_path) + ".childpid").exists()
    assert not Path(str(flash_path) + ".childpid").exists()
    with database.session() as connection:
        run = DeploymentRunRepository(connection).get(
            captured.value.run_id or ""
        )
        members = DeploymentRunRepository(connection).members(
            captured.value.run_id or ""
        )
    assert run is not None and run.status == "failed"
    assert run.failure_kind == "device_capability_mismatch"
    assert members == ()


def test_device_mapping_change_fails_closed(
    tmp_path: Path,
) -> None:
    database, placement_id, inputs, _, _ = _seed(tmp_path)

    with pytest.raises(DeploymentExecutionError) as captured:
        _executor(
            database,
            device_inventory=_inventory(
                vulkan_key="pci:0000:03:00.0"
            ),
        ).execute(
            placement_id,
            inputs,
            readiness_timeout_seconds=1.0,
        )

    assert captured.value.failure_kind == "device_capability_mismatch"
    assert "changed after planning" in str(captured.value)


def test_missing_runtime_gpu_telemetry_fails_closed(
    tmp_path: Path,
) -> None:
    database, placement_id, inputs, _, _ = _seed(tmp_path)
    provider = StaticGpuProvider(
        GpuTelemetrySample(
            device="0000:01:00.0",
            stable_device_key="pci:0000:01:00.0",
            vram_total_bytes=1000,
            vram_used_bytes=800,
        )
    )

    with pytest.raises(DeploymentExecutionError) as captured:
        _executor(database, provider=provider).execute(
            placement_id,
            inputs,
        )

    assert captured.value.failure_kind == "telemetry_incomplete"
    with database.session() as connection:
        run = DeploymentRunRepository(connection).get(
            captured.value.run_id or ""
        )
    assert run is not None
    assert run.failure_kind == "telemetry_incomplete"
    assert run.failure_details is not None
    runtime = run.failure_details["runtime_memory"]
    assert runtime["missing_devices"] == ["pci:0000:02:00.0"]


def test_runtime_projection_overrun_is_persisted_without_margin_failure(
    tmp_path: Path,
) -> None:
    database, placement_id, inputs, _, _ = _seed(tmp_path)

    summary = _executor(
        database,
        provider=_provider(free0=350, free1=500),
    ).execute(
        placement_id,
        inputs,
        residency_hold_seconds=0.01,
    )

    assert summary.status == "completed"
    overruns = summary.runtime_memory["projection_overruns"]
    assert len(overruns) == 1
    assert overruns[0]["device_id"] == "pci:0000:01:00.0"
    assert overruns[0]["projected_bytes"] == 600
    assert overruns[0]["observed_used_bytes"] == 650
    assert overruns[0]["used_delta_bytes"] == 50
    assert summary.runtime_memory["violations"] == []


def test_runtime_margin_violation_is_persisted(tmp_path: Path) -> None:
    database, placement_id, inputs, _, _ = _seed(
        tmp_path,
        margin_bytes=150,
    )

    with pytest.raises(DeploymentExecutionError) as captured:
        _executor(
            database,
            provider=_provider(free0=100, free1=200),
        ).execute(placement_id, inputs)

    assert (
        captured.value.failure_kind
        == "runtime_memory_margin_violated"
    )
    with database.session() as connection:
        run = DeploymentRunRepository(connection).get(
            captured.value.run_id or ""
        )
    assert run is not None
    assert run.failure_kind == "runtime_memory_margin_violated"
    assert run.failure_details is not None
    runtime = run.failure_details["runtime_memory"]
    assert runtime["violations"][0]["device_id"] == "pci:0000:01:00.0"


def test_stale_run_recovery_marks_previous_run_failed(tmp_path: Path) -> None:
    database, placement_id, inputs, _, _ = _seed(tmp_path)
    with database.session() as connection:
        placement = DeploymentPlacementRepository(connection).get(placement_id)
        assert placement is not None
        runs = DeploymentRunRepository(connection)
        stale_id = runs.create(
            deployment_candidate_id=placement.deployment_candidate_id,
            deployment_placement_id=placement_id,
            status="ready",
            started_at="2026-10-05T12:00:00Z",
        )
        runs.add_member(
            stale_id,
            instance_id="qwen",
            endpoint="http://127.0.0.1:1",
            member_status="ready",
        )

    summary = _executor(database).execute(placement_id, inputs)

    with database.session() as connection:
        stale = DeploymentRunRepository(connection).get(stale_id)
        stale_members = DeploymentRunRepository(connection).members(stale_id)
    assert summary.status == "completed"
    assert stale is not None and stale.status == "failed"
    assert stale.failure_kind == "member_crash"
    assert stale_members[0].member_status == "failed"
    assert "stale" in (stale_members[0].cleanup_error or "")
