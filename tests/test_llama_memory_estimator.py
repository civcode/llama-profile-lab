"""Structured memory-estimator helper adapter tests."""

import json
from pathlib import Path

import pytest

from llama_profile_lab.domain import (
    Candidate,
    ComputeConfig,
    ContextConfig,
    FitConfig,
    ModelSelection,
    PlacementConfig,
)
from llama_profile_lab.llama import (
    CapabilitySet,
    MemoryEstimatorAdapter,
    MemoryEstimatorConfigurationError,
    MemoryEstimatorParseError,
)


_REQUIRED = (
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
    "--flash-attn",
)


def capabilities(*extra: str) -> CapabilitySet:
    return CapabilitySet(
        kind="llama-memory-estimator",
        options=frozenset((*_REQUIRED, *extra)),
        help_stdout="",
        help_stderr="",
        help_exit_code=0,
    )


def candidate(
    *,
    context: int = 131072,
    cache_type_k: str = "f16",
    batch_size: int = 4096,
) -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id="model:qwen"),
        context=ContextConfig(
            size=context,
            cache_type_k=cache_type_k,
            cache_type_v="f16",
        ),
        compute=ComputeConfig(
            flash_attn="on",
            batch_size=batch_size,
            ubatch_size=2048,
        ),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=256, min_context=4096),
        ),
    )


def test_build_invocation_preserves_selected_device_order(tmp_path: Path) -> None:
    invocation = MemoryEstimatorAdapter().build_invocation(
        binary_path=tmp_path / "llama-memory-estimator",
        capabilities=capabilities(),
        helper_sha256="a" * 64,
        model_path=tmp_path / "model.gguf",
        candidate=candidate(),
        selected_devices=("CUDA0", "Vulkan0"),
    )

    assert "--device" in invocation.argv
    index = invocation.argv.index("--device")
    assert invocation.argv[index + 1] == "CUDA0,Vulkan0"
    assert invocation.identity.selected_devices == ("CUDA0", "Vulkan0")


def test_cache_identity_changes_with_memory_relevant_candidate_fields(
    tmp_path: Path,
) -> None:
    adapter = MemoryEstimatorAdapter()
    kwargs = {
        "binary_path": tmp_path / "llama-memory-estimator",
        "capabilities": capabilities(),
        "helper_sha256": "a" * 64,
        "model_path": tmp_path / "model.gguf",
        "selected_devices": ("CUDA0", "Vulkan0"),
    }

    base = adapter.build_invocation(candidate=candidate(), **kwargs).identity
    changed_context = adapter.build_invocation(
        candidate=candidate(context=262144),
        **kwargs,
    ).identity
    changed_kv = adapter.build_invocation(
        candidate=candidate(cache_type_k="q8_0"),
        **kwargs,
    ).identity
    changed_batch = adapter.build_invocation(
        candidate=candidate(batch_size=8192),
        **kwargs,
    ).identity
    reversed_devices = adapter.build_invocation(
        candidate=candidate(),
        selected_devices=("Vulkan0", "CUDA0"),
        **{key: value for key, value in kwargs.items() if key != "selected_devices"},
    ).identity

    assert len(
        {
            base.content_hash(),
            changed_context.content_hash(),
            changed_kv.content_hash(),
            changed_batch.content_hash(),
            reversed_devices.content_hash(),
        }
    ) == 5


def test_missing_required_helper_option_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MemoryEstimatorConfigurationError, match="required options"):
        MemoryEstimatorAdapter().build_invocation(
            binary_path=tmp_path / "helper",
            capabilities=CapabilitySet(
                kind="llama-memory-estimator",
                options=frozenset({"--json"}),
                help_stdout="",
                help_stderr="",
                help_exit_code=0,
            ),
            helper_sha256="a" * 64,
            model_path=tmp_path / "model.gguf",
            candidate=candidate(),
        )


def test_parse_structured_memory_output() -> None:
    raw = {
        "schema": "llama-memory-estimate",
        "version": 1,
        "devices": [
            {
                "logical_device_name": "CUDA0",
                "model_bytes": 100,
                "context_bytes": 20,
                "compute_bytes": 5,
                "total_bytes": 125,
                "device_total_bytes": 1000,
                "device_free_bytes": 900,
            },
            {
                "logical_device_name": "Vulkan0",
                "model_bytes": 80,
                "context_bytes": 15,
                "compute_bytes": 5,
                "total_bytes": 100,
                "device_total_bytes": 1000,
                "device_free_bytes": 850,
            },
        ],
        "resolved": {
            "n_gpu_layers": 48,
            "devices": ["CUDA0", "Vulkan0"],
            "split_mode": "layer",
            "main_gpu": 0,
            "tensor_split": [3.0, 1.0],
            "override_tensor": [],
        },
        "metadata": {"source": "synthetic"},
    }

    result = MemoryEstimatorAdapter().parse_output(json.dumps(raw))

    assert result.devices[0].model_bytes == 100
    assert result.resolved.devices == ("CUDA0", "Vulkan0")


@pytest.mark.parametrize(
    "stdout",
    (
        "not-json",
        json.dumps({"schema": "llama-memory-estimate", "version": 1}),
    ),
)
def test_malformed_or_partial_json_is_rejected(stdout: str) -> None:
    with pytest.raises(MemoryEstimatorParseError):
        MemoryEstimatorAdapter().parse_output(stdout)
