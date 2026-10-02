"""llama-server adapter tests."""

from pathlib import Path

import pytest

from llama_profile_lab.domain import (
    Candidate,
    ComputeConfig,
    ContextConfig,
    ModelSelection,
    PlacementConfig,
    ResolvedPlacement,
    ServerConfig,
    SpeculativeConfig,
)
from llama_profile_lab.llama import CapabilitySet, LlamaServerAdapter
from llama_profile_lab.llama.server import LlamaServerConfigurationError


def capabilities() -> CapabilitySet:
    options = {
        "--model",
        "--host",
        "--port",
        "--ctx-size",
        "--batch-size",
        "--ubatch-size",
        "--cache-type-k",
        "--cache-type-v",
        "--parallel",
        "--n-gpu-layers",
        "--kv-offload",
        "--no-kv-offload",
        "--kv-unified",
        "--no-kv-unified",
        "--op-offload",
        "--no-op-offload",
        "--repack",
        "--no-repack",
        "--alias",
        "--spec-type",
        "--spec-draft-n-max",
        "--spec-draft-model",
    }
    return CapabilitySet(
        kind="llama-server",
        options=frozenset(options),
        help_stdout="",
        help_stderr="",
        help_exit_code=0,
    )


def candidate(*, speculative: bool) -> Candidate:
    return Candidate(
        model=ModelSelection(
            target_model_id="model:target",
            draft_model_id="model:draft" if speculative else None,
        ),
        context=ContextConfig(
            size=8192,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        ),
        compute=ComputeConfig(
            batch_size=2048,
            ubatch_size=512,
        ),
        placement=PlacementConfig(mode="fixed"),
        server=ServerConfig(parallel=1),
        speculative=(
            SpeculativeConfig(enabled=True, type="draft-mtp", draft_n_max=3)
            if speculative
            else SpeculativeConfig()
        ),
    )


def placement() -> ResolvedPlacement:
    return ResolvedPlacement(
        production_context_size=8192,
        n_gpu_layers=42,
        n_cpu_moe=0,
        split_mode="layer",
        main_gpu=0,
        devices="auto",
        tensor_split=None,
        override_tensor=(),
    )


def test_server_argv_freezes_placement_and_speculative_configuration() -> None:
    argv = LlamaServerAdapter().build_argv(
        binary_path=Path("/opt/llama-server"),
        capabilities=capabilities(),
        model_path=Path("/models/target.gguf"),
        draft_model_path=Path("/models/draft.gguf"),
        candidate=candidate(speculative=True),
        placement=placement(),
        host="127.0.0.1",
        port=8080,
        model_alias="spec",
    )

    assert argv[0] == "/opt/llama-server"
    assert ("--ctx-size", "8192") == (argv[argv.index("--ctx-size")], argv[argv.index("--ctx-size") + 1])
    assert "--n-gpu-layers" in argv
    assert argv[argv.index("--n-gpu-layers") + 1] == "42"
    assert argv[argv.index("--spec-type") + 1] == "draft-mtp"
    assert argv[argv.index("--spec-draft-n-max") + 1] == "3"
    assert argv[argv.index("--spec-draft-model") + 1] == "/models/draft.gguf"
    assert "--fit-target" not in argv


def test_server_requires_draft_path_when_candidate_references_draft_model() -> None:
    with pytest.raises(LlamaServerConfigurationError, match="draft model path"):
        LlamaServerAdapter().build_argv(
            binary_path=Path("/opt/llama-server"),
            capabilities=capabilities(),
            model_path=Path("/models/target.gguf"),
            candidate=candidate(speculative=True),
            placement=placement(),
            host="127.0.0.1",
            port=8080,
        )
