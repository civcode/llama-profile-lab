"""Candidate-dependent workload expansion tests."""

from llama_profile_lab.domain import (
    AbsoluteDepth,
    Candidate,
    ComputeConfig,
    ContextConfig,
    DecodeSuiteCase,
    FitConfig,
    FractionalDepth,
    ModelSelection,
    PlacementConfig,
    PrefillSuiteCase,
    WorkloadSuite,
)
from llama_profile_lab.domain.workload import DecodeWorkloadCase
from llama_profile_lab.planning import expand_workload_suite


def candidate(context_size: int = 131072) -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id="model:sha256:flash"),
        context=ContextConfig(
            size=context_size,
            cache_type_k="f16",
            cache_type_v="f16",
        ),
        compute=ComputeConfig(batch_size=4096, ubatch_size=2048),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=256, min_context=4096),
        ),
    )


def test_fractional_decode_depth_matches_reference_value() -> None:
    suite = WorkloadSuite(
        id="depth-test",
        cases=(
            DecodeSuiteCase(
                label="tg-mid",
                generate_tokens=256,
                depth=FractionalDepth(value=0.5),
            ),
        ),
    )

    expanded = expand_workload_suite(candidate(), suite)
    case = expanded[0].case

    assert isinstance(case, DecodeWorkloadCase)
    assert case.depth_tokens == 65408
    assert expanded[0].provenance["candidate_context_size"] == 131072


def test_relative_depth_changes_with_candidate_context() -> None:
    suite = WorkloadSuite(
        id="depth-test",
        cases=(
            DecodeSuiteCase(
                generate_tokens=256,
                depth=FractionalDepth(value=0.5),
            ),
        ),
    )

    first = expand_workload_suite(candidate(65536), suite)[0].case
    second = expand_workload_suite(candidate(131072), suite)[0].case

    assert first.content_hash() != second.content_hash()


def test_absolute_prefill_is_preserved() -> None:
    suite = WorkloadSuite(
        id="prefill",
        cases=(
            PrefillSuiteCase(
                prompt_tokens=8192,
                depth=AbsoluteDepth(tokens=0),
            ),
        ),
    )

    expanded = expand_workload_suite(candidate(), suite)

    assert expanded[0].case.prompt_tokens == 8192
    assert expanded[0].case.depth_tokens == 0
