"""llama-fit-params argv and output parsing tests."""

from pathlib import Path

import pytest

from llama_profile_lab.domain import (
    Candidate,
    ComputeConfig,
    ContextConfig,
    FitConfig,
    ModelSelection,
    PlacementConfig,
    PlacementConstraints,
)
from llama_profile_lab.llama import CapabilitySet
from llama_profile_lab.llama.fit_params import (
    LlamaFitParamsAdapter,
    LlamaFitParamsConfigurationError,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "llama"


def candidate() -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id="model:sha256:flash"),
        context=ContextConfig(
            size=131072,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        ),
        compute=ComputeConfig(
            flash_attn="on",
            batch_size=4096,
            ubatch_size=2048,
            load_mode="mmap",
            lazy_mode="on",
        ),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=256, min_context=4096),
        ),
    )


def capabilities() -> CapabilitySet:
    return CapabilitySet(
        kind="llama-fit-params",
        options=frozenset(
            {
                "--model",
                "--ctx-size",
                "--batch-size",
                "--ubatch-size",
                "--cache-type-k",
                "--cache-type-v",
                "--flash-attn",
                "--load-mode",
                "--lazy-mode",
                "--fit-target",
                "--fit-ctx",
                "--kv-offload",
                "--no-kv-offload",
                "--op-offload",
                "--no-op-offload",
                "--no-host",
                "--repack",
                "--no-repack",
            }
        ),
        help_stdout="",
        help_stderr="",
        help_exit_code=0,
    )


def test_fit_argv_pins_full_production_context() -> None:
    argv = LlamaFitParamsAdapter().build_argv(
        binary_path=Path("/bin/llama-fit-params"),
        capabilities=capabilities(),
        model_path=Path("/models/flash.gguf"),
        candidate=candidate(),
    )

    assert argv[0] == "/bin/llama-fit-params"
    assert argv[argv.index("--ctx-size") + 1] == "131072"
    assert argv[argv.index("--fit-target") + 1] == "256"
    assert argv[argv.index("--fit-ctx") + 1] == "4096"
    assert "--n-gpu-layers" not in argv
    assert "--kv-offload" in argv
    assert "--op-offload" in argv
    assert "--repack" in argv
    assert ("--repack", "1") not in tuple(zip(argv, argv[1:]))


def test_parser_extracts_concrete_placement() -> None:
    output = (_FIXTURES / "llama-fit-params-output.txt").read_text(encoding="utf-8")

    result = LlamaFitParamsAdapter().parse_output(output, candidate=candidate())

    assert result.placement.production_context_size == 131072
    assert result.placement.n_gpu_layers == 42
    assert result.placement.tensor_split == (3.0, 1.0)
    assert result.placement.override_tensor == (
        r"blk\.12\.ffn_.*=CPU,blk\.13\.ffn_.*=CPU",
    )


def test_fit_rejects_fixed_gpu_layers() -> None:
    base = candidate()
    payload = base.model_dump(mode="python")
    payload["placement"]["constraints"] = PlacementConstraints(
        n_gpu_layers=32
    ).model_dump(mode="python")
    fixed = Candidate.model_validate(payload)

    with pytest.raises(LlamaFitParamsConfigurationError, match="n_gpu_layers"):
        LlamaFitParamsAdapter().build_argv(
            binary_path=Path("/bin/llama-fit-params"),
            capabilities=capabilities(),
            model_path=Path("/models/flash.gguf"),
            candidate=fixed,
        )



def test_fit_argv_uses_negative_common_flags_when_disabled() -> None:
    base = candidate()
    payload = base.model_dump(mode="python")
    payload["context"]["kv_offload"] = False
    payload["compute"]["no_op_offload"] = True
    payload["compute"]["no_host"] = True
    payload["compute"]["repack"] = False
    modified = Candidate.model_validate(payload)

    argv = LlamaFitParamsAdapter().build_argv(
        binary_path=Path("/bin/llama-fit-params"),
        capabilities=capabilities(),
        model_path=Path("/models/flash.gguf"),
        candidate=modified,
    )

    assert "--no-kv-offload" in argv
    assert "--no-op-offload" in argv
    assert "--no-host" in argv
    assert "--no-repack" in argv



def test_parser_preserves_repeated_override_tensor_arguments() -> None:
    output = (
        "-c 131072 -ngl 42 "
        "-ot 'blk\\.12\\..*=CPU' "
        "-ot 'blk\\.13\\..*=CPU'\n"
    )

    result = LlamaFitParamsAdapter().parse_output(output, candidate=candidate())

    assert result.placement.override_tensor == (
        r"blk\.12\..*=CPU",
        r"blk\.13\..*=CPU",
    )
