"""Deployment search-space domain tests."""

import pytest
from pydantic import ValidationError

from llama_profile_lab.domain import (
    DeploymentSearchDimension,
    DeploymentSearchSpace,
)


def test_deployment_search_accepts_tuple_placement_values() -> None:
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.requested_placement.devices",
                values=(("CUDA0",), ("CUDA0", "Vulkan0")),
            ),
            DeploymentSearchDimension(
                path="instances.qwen.requested_placement.tensor_split",
                values=((1.0,), (3.0, 1.0)),
            ),
        )
    )

    assert search.dimensions[0].values[1] == ("CUDA0", "Vulkan0")


def test_deployment_search_rejects_duplicate_paths_and_values() -> None:
    with pytest.raises(ValidationError, match="values must be unique"):
        DeploymentSearchDimension(
            path="instances.qwen.context.size",
            values=(4096, 4096),
        )

    dimension = DeploymentSearchDimension(
        path="instances.qwen.context.size",
        values=(4096,),
    )
    with pytest.raises(ValidationError, match="paths must be unique"):
        DeploymentSearchSpace(dimensions=(dimension, dimension))
