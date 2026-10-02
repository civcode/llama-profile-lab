"""N-dimensional Candidate expansion tests."""

from llama_profile_lab.domain import (
    Candidate,
    ComputeConfig,
    ContextConfig,
    FitConfig,
    ModelSelection,
    PlacementConfig,
    SearchDimension,
    SearchSpace,
)
from llama_profile_lab.planning import expand_search_space


def base_candidate() -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id="model:sha256:flash"),
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


def test_reference_batch_ubatch_search_produces_eleven_candidates() -> None:
    search = SearchSpace(
        dimensions=(
            SearchDimension(
                path="compute.batch_size",
                values=(2048, 4096, 8192),
            ),
            SearchDimension(
                path="compute.ubatch_size",
                values=(512, 1024, 2048, 4096),
            ),
        ),
        constraints=("compute.ubatch_size <= compute.batch_size",),
    )

    expansion = expand_search_space(base_candidate(), search)

    assert expansion.raw_combinations == 12
    assert expansion.rejected_by_constraints == 1
    assert expansion.duplicate_candidates == 0
    assert len(expansion.candidates) == 11

    pairs = [
        (
            point.candidate.compute.batch_size,
            point.candidate.compute.ubatch_size,
        )
        for point in expansion.candidates
    ]
    assert pairs == [
        (2048, 512),
        (2048, 1024),
        (2048, 2048),
        (4096, 512),
        (4096, 1024),
        (4096, 2048),
        (4096, 4096),
        (8192, 512),
        (8192, 1024),
        (8192, 2048),
        (8192, 4096),
    ]


def test_conditional_dimension_does_not_multiply_when_disabled() -> None:
    search = SearchSpace(
        dimensions=(
            SearchDimension(path="speculative.enabled", values=(False, True)),
            SearchDimension(
                path="speculative.type",
                values=("draft-mtp",),
                condition="speculative.enabled == true",
            ),
            SearchDimension(
                path="speculative.draft_n_max",
                values=(1, 2, 3),
                condition="speculative.enabled == true",
            ),
        )
    )
    candidate = base_candidate().model_copy(
        update={
            "speculative": {
                "enabled": False,
                "type": None,
                "draft_n_max": None,
            }
        }
    )

    # model_copy does not recursively validate updates, so revalidate the payload.
    candidate = Candidate.model_validate(candidate.model_dump(mode="python"))
    expansion = expand_search_space(candidate, search)

    assert expansion.raw_combinations == 4
    assert len(expansion.candidates) == 4
