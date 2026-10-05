"""Concurrent workload domain and phase generation tests."""

import pytest
from pydantic import ValidationError

from llama_profile_lab.domain import (
    AbsoluteDepth,
    Candidate,
    ComputeConfig,
    ContextConfig,
    DecodeSuiteCase,
    DeploymentCandidate,
    DeploymentWorkloadMix,
    FitConfig,
    HostResourcePolicy,
    ModelInstanceCandidate,
    ModelSelection,
    PlacementConfig,
    PrefillSuiteCase,
    WorkloadSuite,
)
from llama_profile_lab.planning import generate_concurrent_workloads


def _candidate(model_id: str, context: int = 4096) -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id=model_id),
        context=ContextConfig(size=context, cache_type_k="f16", cache_type_v="f16"),
        compute=ComputeConfig(batch_size=512, ubatch_size=128),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=0, min_context=1024),
        ),
    )


def _deployment() -> DeploymentCandidate:
    return DeploymentCandidate(
        instances=(
            ModelInstanceCandidate(
                instance_id="a",
                candidate_id="cand_a",
                role="primary",
                model_artifact_id="model:a",
                binary_id="bin_server",
                server_identity="a",
            ),
            ModelInstanceCandidate(
                instance_id="b",
                candidate_id="cand_b",
                role="secondary",
                model_artifact_id="model:b",
                binary_id="bin_server",
                server_identity="b",
            ),
        ),
        resource_policy=HostResourcePolicy(),
        workload_mix=DeploymentWorkloadMix(
            workload_suite_id="suite_repository_id"
        ),
    )


def test_phase_generation_covers_equal_and_asymmetric_depths() -> None:
    suite = WorkloadSuite(
        id="authored-suite-name",
        cases=(
            PrefillSuiteCase(
                prompt_tokens=64,
                depth=AbsoluteDepth(tokens=128),
            ),
            PrefillSuiteCase(
                prompt_tokens=64,
                depth=AbsoluteDepth(tokens=1024),
            ),
            DecodeSuiteCase(
                generate_tokens=32,
                depth=AbsoluteDepth(tokens=128),
            ),
            DecodeSuiteCase(
                generate_tokens=32,
                depth=AbsoluteDepth(tokens=1024),
            ),
        ),
    )

    cases = generate_concurrent_workloads(
        _deployment(),
        candidates={
            "cand_a": _candidate("model:a"),
            "cand_b": _candidate("model:b"),
        },
        suite=suite,
    )

    assert len(cases) == 16
    assert {case.phase for case in cases} == {"dd", "pp", "pd", "dp"}

    pp_depths = {
        tuple(member.depth_tokens for member in case.members)
        for case in cases
        if case.phase == "pp"
    }
    assert pp_depths == {
        (128, 128),
        (128, 1024),
        (1024, 128),
        (1024, 1024),
    }


def test_concurrent_phase_mode_mismatch_is_rejected() -> None:
    from llama_profile_lab.domain import (
        ConcurrentWorkloadCase,
        ConcurrentWorkloadMemberSpec,
    )

    with pytest.raises(ValidationError, match="phase dd"):
        ConcurrentWorkloadCase(
            phase="dd",
            members=(
                ConcurrentWorkloadMemberSpec(
                    ordinal=0,
                    instance_id="a",
                    mode="prefill",
                    prompt_tokens=64,
                ),
                ConcurrentWorkloadMemberSpec(
                    ordinal=1,
                    instance_id="b",
                    mode="decode",
                    generate_tokens=32,
                ),
            ),
        )
