"""Candidate schema validation tests."""

import pytest
from pydantic import ValidationError

from llama_profile_lab.domain import (
    Candidate,
    ComputeConfig,
    ContextConfig,
    FitConfig,
    ModelSelection,
    PlacementConfig,
    SpeculativeConfig,
)


def test_ubatch_cannot_exceed_batch() -> None:
    with pytest.raises(ValidationError, match="ubatch_size"):
        ComputeConfig(batch_size=2048, ubatch_size=4096)


def test_fit_mode_requires_fit_policy() -> None:
    with pytest.raises(ValidationError, match="requires a fit policy"):
        PlacementConfig(mode="fit")


def test_fixed_mode_rejects_fit_policy() -> None:
    with pytest.raises(ValidationError, match="cannot include a fit policy"):
        PlacementConfig(
            mode="fixed",
            fit=FitConfig(target_mib=256, min_context=4096),
        )


def test_disabled_speculation_rejects_draft_settings() -> None:
    with pytest.raises(ValidationError, match="disabled speculative"):
        SpeculativeConfig(enabled=False, type="draft-mtp", draft_n_max=3)


def test_enabled_speculation_requires_type_and_draft_count() -> None:
    with pytest.raises(ValidationError, match="requires a type"):
        SpeculativeConfig(enabled=True, draft_n_max=3)


def test_candidate_serializes_extra_args_as_object() -> None:
    candidate = Candidate(
        model=ModelSelection(target_model_id="model:sha256:abc"),
        context=ContextConfig(
            size=65536,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        ),
        compute=ComputeConfig(batch_size=4096, ubatch_size=2048),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=256, min_context=4096),
        ),
        extra_args={
            "--flag": True,
            "-pg": (512, 128),
        },
    )

    dumped = candidate.model_dump(mode="json")
    assert dumped["extra_args"] == {
        "--flag": True,
        "-pg": [512, 128],
    }
