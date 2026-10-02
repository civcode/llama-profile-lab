"""Concrete placement produced by placement resolution."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt

from llama_profile_lab.domain.base import FrozenModel

NonEmptyString = Annotated[str, Field(min_length=1)]
type ResolvedDeviceSetting = Literal["auto"] | tuple[NonEmptyString, ...]


class ResolvedPlacement(FrozenModel):
    """Concrete placement arguments safe to reuse across benchmark workloads."""

    production_context_size: Annotated[int, Field(gt=0)]
    n_gpu_layers: NonNegativeInt
    n_cpu_moe: NonNegativeInt = 0
    split_mode: NonEmptyString = "layer"
    main_gpu: NonNegativeInt = 0
    devices: ResolvedDeviceSetting = "auto"
    tensor_split: tuple[Annotated[float, Field(ge=0)], ...] | None = None
    override_tensor: tuple[NonEmptyString, ...] = ()
