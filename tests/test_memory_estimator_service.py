"""Integration tests for V2 device inventory and memory estimation."""

from __future__ import annotations

import stat
from pathlib import Path
from threading import Event

import pytest

from llama_profile_lab.db import (
    AcceleratorDeviceRepository,
    CandidateRepository,
    Database,
    EnvironmentRepository,
    MemoryEstimateRepository,
)
from llama_profile_lab.domain import (
    Candidate,
    ComputeConfig,
    ContextConfig,
    FitConfig,
    ModelSelection,
    PlacementConfig,
)
from llama_profile_lab.execution import (
    DeviceInventoryService,
    MemoryEstimatorError,
    MemoryEstimatorService,
)
from llama_profile_lab.llama import BinaryProbe, probe_binary


_HELP_OPTIONS = (
    "--json",
    "--model",
    "--ctx-size",
    "--batch-size",
    "--ubatch-size",
    "--cache-type-k",
    "--cache-type-v",
    "--gpu-layers",
    "--split-mode",
    "--main-gpu",
    "--device",
    "--tensor-split",
    "--override-tensor",
    "--flash-attn",
    "--load-mode",
    "--lazy-mode",
    "--no-kv-offload",
    "--no-op-offload",
    "--no-host",
    "--no-repack",
    "--list-devices",
)


def make_candidate() -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id="model:qwen"),
        context=ContextConfig(
            size=131072,
            cache_type_k="f16",
            cache_type_v="f16",
        ),
        compute=ComputeConfig(
            flash_attn="on",
            batch_size=4096,
            ubatch_size=2048,
        ),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=256, min_context=4096),
        ),
    )


def write_helper(path: Path) -> None:
    options = repr("\n".join(_HELP_OPTIONS))
    path.write_text(
        f"""#!/usr/bin/env python3
import json
import sys
import time

OPTIONS = {options}
args = sys.argv[1:]

if "--version" in args:
    print("version 1 commit abcdef1")
    raise SystemExit(0)

if "--help" in args:
    print(OPTIONS)
    raise SystemExit(0)

if "--list-devices" in args:
    print("Available devices:")
    print("  CUDA0: NVIDIA Test GPU (16384 MiB, 15000 MiB free)")
    print("  Vulkan0: AMD Test GPU (16384 MiB, 14000 MiB free)")
    raise SystemExit(0)

model = args[args.index("--model") + 1]
if model.endswith("fail.gguf"):
    print("synthetic helper failure", file=sys.stderr)
    raise SystemExit(7)
if model.endswith("badjson.gguf"):
    print("{{not-json")
    raise SystemExit(0)
if model.endswith("sleep.gguf"):
    time.sleep(2)

if "--device" in args:
    devices = args[args.index("--device") + 1].split(",")
else:
    devices = ["CUDA0", "Vulkan0"]

rows = []
for index, name in enumerate(devices):
    model_bytes = 100 + index * 10
    context_bytes = 20 + index * 5
    compute_bytes = 5
    rows.append({{
        "logical_device_name": name,
        "model_bytes": model_bytes,
        "context_bytes": context_bytes,
        "compute_bytes": compute_bytes,
        "total_bytes": model_bytes + context_bytes + compute_bytes,
        "device_total_bytes": 1000,
        "device_free_bytes": 800 - index * 10,
    }})

print(json.dumps({{
    "schema": "llama-memory-estimate",
    "version": 1,
    "devices": rows,
    "resolved": {{
        "n_gpu_layers": 48,
        "devices": devices,
        "split_mode": "layer",
        "main_gpu": 0,
        "tensor_split": None,
        "override_tensor": [],
    }},
    "metadata": {{"source": "synthetic"}},
}}))
print("synthetic helper warning", file=sys.stderr)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def register_helper(database: Database, path: Path) -> tuple[str, BinaryProbe]:
    probe = probe_binary(path, kind="llama-memory-estimator")
    with database.session() as connection:
        binary_id = EnvironmentRepository(connection).put_binary(
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
    return binary_id, probe


def seed(tmp_path: Path) -> tuple[Database, str, str, Path]:
    database = Database(tmp_path / "m2.db")
    helper_path = tmp_path / "llama-memory-estimator"
    write_helper(helper_path)
    helper_id, _ = register_helper(database, helper_path)
    with database.session() as connection:
        candidate_id = CandidateRepository(connection).put(make_candidate())
    return database, candidate_id, helper_id, helper_path


def test_device_inventory_and_memory_estimate_cache(tmp_path: Path) -> None:
    database, candidate_id, helper_id, _ = seed(tmp_path)

    inventory = DeviceInventoryService(database).inspect(helper_id)
    assert [item.logical_device_name for item in inventory.devices] == [
        "CUDA0",
        "Vulkan0",
    ]

    with database.session() as connection:
        persisted = AcceleratorDeviceRepository(connection).list_for_binary(
            host_id=inventory.host_id,
            binary_id=helper_id,
        )
        assert [item.logical_device_name for item in persisted] == [
            "CUDA0",
            "Vulkan0",
        ]
        assert all(item.mapping_status == "unresolved" for item in persisted)

        AcceleratorDeviceRepository(connection).put_inventory(
            host_id=inventory.host_id,
            binary_id=helper_id,
            devices=inventory.devices[:1],
            raw_output="refreshed",
        )
        refreshed = AcceleratorDeviceRepository(connection).list_for_binary(
            host_id=inventory.host_id,
            binary_id=helper_id,
        )
        assert [item.logical_device_name for item in refreshed] == ["CUDA0"]

    service = MemoryEstimatorService(database)
    first = service.estimate(
        candidate_id,
        helper_binary_id=helper_id,
        model_path=tmp_path / "model.gguf",
        selected_devices=("CUDA0", "Vulkan0"),
    )

    assert first.cache_hit is False
    assert first.output.resolved.devices == ("CUDA0", "Vulkan0")
    with database.session() as connection:
        repository = MemoryEstimateRepository(connection)
        attempt = repository.attempt(first.attempt_id)
        assert attempt is not None
        assert attempt.status == "completed"
        assert "llama-memory-estimate" in attempt.stdout
        assert "synthetic helper warning" in attempt.stderr
        rows = repository.devices(first.estimate_id)
        assert [row.logical_device_name for row in rows] == [
            "CUDA0",
            "Vulkan0",
        ]
        assert [row.ordinal for row in rows] == [0, 1]

    second = service.estimate(
        candidate_id,
        helper_binary_id=helper_id,
        model_path=tmp_path / "model.gguf",
        selected_devices=("CUDA0", "Vulkan0"),
    )
    assert second.cache_hit is True
    assert second.estimate_id == first.estimate_id
    assert second.attempt_id == first.attempt_id


def test_helper_hash_drift_is_rejected_before_cache_use(tmp_path: Path) -> None:
    database, candidate_id, helper_id, helper_path = seed(tmp_path)
    service = MemoryEstimatorService(database)
    first = service.estimate(
        candidate_id,
        helper_binary_id=helper_id,
        model_path=tmp_path / "model.gguf",
        selected_devices=("CUDA0", "Vulkan0"),
    )
    assert first.cache_hit is False

    with helper_path.open("a", encoding="utf-8") as handle:
        handle.write("\n# drift\n")

    with pytest.raises(MemoryEstimatorError, match="changed on disk"):
        service.estimate(
            candidate_id,
            helper_binary_id=helper_id,
            model_path=tmp_path / "model.gguf",
            selected_devices=("CUDA0", "Vulkan0"),
        )

    with database.session() as connection:
        row = connection.execute(
            """
            SELECT id
            FROM memory_estimate_attempt
            WHERE status = 'binary_changed'
            ORDER BY started_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        assert row is not None
        attempt = MemoryEstimateRepository(connection).attempt(str(row["id"]))
        assert attempt is not None
        assert attempt.status == "binary_changed"
        assert attempt.failure_details is not None
        assert attempt.failure_details["expected_sha256"] != (
            attempt.failure_details["actual_sha256"]
        )


@pytest.mark.parametrize(
    ("model_name", "status", "error"),
    (
        ("fail.gguf", "failed", "exited with code 7"),
        ("badjson.gguf", "parser_failed", "valid JSON"),
        ("sleep.gguf", "timeout", "timed out"),
    ),
)
def test_estimator_failures_are_persisted(
    tmp_path: Path,
    model_name: str,
    status: str,
    error: str,
) -> None:
    database, candidate_id, helper_id, _ = seed(tmp_path)
    timeout = 0.05 if model_name == "sleep.gguf" else 2.0

    with pytest.raises(MemoryEstimatorError, match=error):
        MemoryEstimatorService(database).estimate(
            candidate_id,
            helper_binary_id=helper_id,
            model_path=tmp_path / model_name,
            selected_devices=("CUDA0", "Vulkan0"),
            timeout_seconds=timeout,
        )

    with database.session() as connection:
        row = connection.execute(
            """
            SELECT id
            FROM memory_estimate_attempt
            ORDER BY started_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        assert row is not None
        attempt = MemoryEstimateRepository(connection).attempt(str(row["id"]))
        assert attempt is not None
        assert attempt.status == status


def test_estimator_cancellation_is_persisted(tmp_path: Path) -> None:
    database, candidate_id, helper_id, _ = seed(tmp_path)
    cancel = Event()
    cancel.set()

    with pytest.raises(MemoryEstimatorError, match="cancelled"):
        MemoryEstimatorService(database).estimate(
            candidate_id,
            helper_binary_id=helper_id,
            model_path=tmp_path / "sleep.gguf",
            selected_devices=("CUDA0", "Vulkan0"),
            timeout_seconds=2.0,
            cancel_event=cancel,
        )

    with database.session() as connection:
        row = connection.execute(
            """
            SELECT id
            FROM memory_estimate_attempt
            ORDER BY started_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        assert row is not None
        attempt = MemoryEstimateRepository(connection).attempt(str(row["id"]))
        assert attempt is not None
        assert attempt.status == "cancelled"
