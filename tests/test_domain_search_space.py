"""Search-space schema tests."""

import pytest
from pydantic import ValidationError

from llama_profile_lab.domain import SearchDimension, SearchSpace


def test_search_space_accepts_arbitrary_dimension_count() -> None:
    space = SearchSpace(
        dimensions=(
            SearchDimension(path="compute.batch_size", values=(2048, 4096, 8192)),
            SearchDimension(path="compute.ubatch_size", values=(512, 1024, 2048)),
            SearchDimension(path="context.cache_type_k", values=("f16", "q8_0")),
        ),
        constraints=("compute.ubatch_size <= compute.batch_size",),
    )

    assert len(space.dimensions) == 3
    assert space.model_dump(mode="json")["constraints"] == [
        "compute.ubatch_size <= compute.batch_size"
    ]


def test_duplicate_dimension_path_is_rejected() -> None:
    with pytest.raises(ValidationError, match="paths must be unique"):
        SearchSpace(
            dimensions=(
                SearchDimension(path="compute.batch_size", values=(2048,)),
                SearchDimension(path="compute.batch_size", values=(4096,)),
            )
        )


def test_duplicate_dimension_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="values must be unique"):
        SearchDimension(path="compute.batch_size", values=(2048, 2048))


def test_condition_is_preserved() -> None:
    dimension = SearchDimension(
        path="speculative.draft_n_max",
        values=(1, 2, 3),
        condition="speculative.enabled == true",
    )
    assert dimension.condition == "speculative.enabled == true"
