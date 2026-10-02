"""llama-bench adapter and JSON parser tests."""

from pathlib import Path

import pytest

from llama_profile_lab.domain import (
    Candidate,
    ComputeConfig,
    ContextConfig,
    DecodeWorkloadCase,
    FitConfig,
    MeasurementPolicy,
    ModelSelection,
    PlacementConfig,
)
from llama_profile_lab.llama import CapabilitySet
from llama_profile_lab.llama.bench import (
    LlamaBenchAdapter,
    LlamaBenchParseError,
    parse_llama_bench_json,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "llama"


def candidate() -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id="model:sha256:fake"),
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


def capabilities() -> CapabilitySet:
    options = frozenset(
        {
            "--model",
            "--n-prompt",
            "--n-gen",
            "--n-depth",
            "--batch-size",
            "--ubatch-size",
            "--cache-type-k",
            "--cache-type-v",
            "--threads",
            "--flash-attn",
            "--load-mode",
            "--lazy-mode",
            "--no-kv-offload",
            "--no-op-offload",
            "--no-host",
            "--repack",
            "--repetitions",
            "--output",
            "--no-warmup",
            "--delay",
        }
    )
    return CapabilitySet(
        kind="llama-bench",
        options=options,
        help_stdout="",
        help_stderr="",
        help_exit_code=0,
    )


def test_adapter_prefers_long_form_and_does_not_apply_fit_target() -> None:
    argv = LlamaBenchAdapter().build_argv(
        binary_path=Path("/bin/llama-bench"),
        capabilities=capabilities(),
        model_path=Path("/models/fake.gguf"),
        candidate=candidate(),
        workload=DecodeWorkloadCase(generate_tokens=256, depth_tokens=4096),
        measurement_policy=MeasurementPolicy(repetitions=3),
    )

    assert "--model" in argv
    assert "--n-gen" in argv
    assert "--n-depth" in argv
    assert "--batch-size" in argv
    assert "--ubatch-size" in argv
    assert "--repetitions" in argv
    assert "--output" in argv
    assert "--fit-target" not in argv
    assert "-n" not in argv


def test_parser_extracts_individual_samples() -> None:
    result = parse_llama_bench_json(
        (_FIXTURES / "llama-bench-result.json").read_text(encoding="utf-8"),
        expected_repetitions=3,
    )

    assert len(result.samples) == 3
    assert result.samples[0].elapsed_ns == 1_000_000_000
    assert result.avg_ts == 256.0


def test_parser_rejects_unexpected_sample_count() -> None:
    text = (_FIXTURES / "llama-bench-result.json").read_text(encoding="utf-8")
    with pytest.raises(LlamaBenchParseError, match="sample count"):
        parse_llama_bench_json(text, expected_repetitions=5)
