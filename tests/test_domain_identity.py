"""Tests for canonical JSON and content identities."""

import pytest
from pydantic import ValidationError

from llama_profile_lab.domain import (
    Candidate,
    ComputeConfig,
    ContextConfig,
    FitConfig,
    ModelSelection,
    PlacementConfig,
    WorkloadSuite,
)
from llama_profile_lab.domain.workload import AbsoluteDepth, PrefillSuiteCase


def make_candidate(*, batch_size: int = 4096) -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id="model:sha256:abc"),
        context=ContextConfig(
            size=131072,
            cache_type_k="f16",
            cache_type_v="f16",
        ),
        compute=ComputeConfig(
            flash_attn="on",
            batch_size=batch_size,
            ubatch_size=1024,
            threads=16,
            load_mode="mmap",
            lazy_mode="on",
        ),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=256, min_context=4096),
        ),
        extra_args={
            "--alpha": 1,
            "--feature": True,
        },
    )


def test_equivalent_candidates_have_identical_identity() -> None:
    first = make_candidate()
    second = Candidate.model_validate(first.model_dump(mode="json"))

    assert first.canonical_identity_json() == second.canonical_identity_json()
    assert first.content_hash() == second.content_hash()


def test_candidate_hash_changes_when_semantic_value_changes() -> None:
    assert make_candidate(batch_size=4096).content_hash() != make_candidate(
        batch_size=8192
    ).content_hash()


def test_extra_args_order_does_not_change_identity() -> None:
    first = make_candidate()
    payload = first.model_dump(mode="json")
    payload["extra_args"] = {
        "--feature": True,
        "--alpha": 1,
    }
    second = Candidate.model_validate(payload)

    assert first.content_hash() == second.content_hash()


def test_frozen_candidate_rejects_assignment() -> None:
    candidate = make_candidate()

    with pytest.raises(ValidationError):
        candidate.compute.batch_size = 8192  # type: ignore[misc]


def test_workload_suite_hash_ignores_presentation_metadata() -> None:
    first = WorkloadSuite(
        id="screen-a",
        description="first description",
        cases=(
            PrefillSuiteCase(
                label="pretty label",
                prompt_tokens=2048,
                depth=AbsoluteDepth(tokens=0),
            ),
        ),
    )
    second = WorkloadSuite(
        id="renamed-suite",
        description="different description",
        cases=(
            PrefillSuiteCase(
                label="renamed case",
                prompt_tokens=2048,
                depth=AbsoluteDepth(tokens=0),
            ),
        ),
    )

    assert first.content_hash() == second.content_hash()
