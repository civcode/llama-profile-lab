"""Central registry of tunable Candidate parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llama_profile_lab.domain.base import JsonScalar
from llama_profile_lab.domain.search_space import SearchSpace

ToolName = Literal["llama-bench", "llama-server", "llama-fit-params", "speed-bench"]


class ParameterError(ValueError):
    """Raised when a search dimension references an invalid parameter."""


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    """Metadata and value validation for one Candidate parameter."""

    path: str
    label: str
    category: str
    python_types: tuple[type[object], ...]
    cli_argument: str | None
    affects_placement: bool
    supported_by: frozenset[ToolName]
    minimum: int | float | None = None
    maximum: int | float | None = None
    string_choices: frozenset[str] | None = None

    def validate_value(self, value: JsonScalar) -> None:
        """Validate one discrete search value."""
        if type(value) not in self.python_types:
            expected = ", ".join(item.__name__ for item in self.python_types)
            raise ParameterError(
                f"{self.path} expects {expected}; got {type(value).__name__}"
            )

        if isinstance(value, bool):
            numeric_value: int | float | None = None
        elif isinstance(value, (int, float)):
            numeric_value = value
        else:
            numeric_value = None

        if (
            numeric_value is not None
            and self.minimum is not None
            and numeric_value < self.minimum
        ):
            raise ParameterError(
                f"{self.path} must be greater than or equal to {self.minimum}"
            )
        if (
            numeric_value is not None
            and self.maximum is not None
            and numeric_value > self.maximum
        ):
            raise ParameterError(
                f"{self.path} must be less than or equal to {self.maximum}"
            )

        if (
            isinstance(value, str)
            and self.string_choices is not None
            and value not in self.string_choices
        ):
            choices = ", ".join(sorted(self.string_choices))
            raise ParameterError(f"{self.path} must be one of: {choices}")


class ParameterRegistry:
    """Lookup and validation for performance-relevant Candidate paths."""

    def __init__(self, definitions: tuple[ParameterDefinition, ...]) -> None:
        by_path = {definition.path: definition for definition in definitions}
        if len(by_path) != len(definitions):
            raise ValueError("parameter definitions must have unique paths")
        self._definitions = by_path

    def get(self, path: str) -> ParameterDefinition:
        """Return a registered parameter definition."""
        try:
            return self._definitions[path]
        except KeyError as exc:
            raise ParameterError(f"unsupported search parameter: {path}") from exc

    def validate_search_space(self, search_space: SearchSpace) -> None:
        """Validate all dimensions against registered parameter metadata."""
        for dimension in search_space.dimensions:
            definition = self.get(dimension.path)
            for value in dimension.values:
                definition.validate_value(value)

    def definitions(self) -> tuple[ParameterDefinition, ...]:
        """Return definitions in deterministic path order."""
        return tuple(self._definitions[path] for path in sorted(self._definitions))


_CACHE_TYPES = frozenset(
    {
        "f32",
        "f16",
        "bf16",
        "q8_0",
        "q4_0",
        "q4_1",
        "iq4_nl",
        "q5_0",
        "q5_1",
    }
)

DEFAULT_PARAMETER_REGISTRY = ParameterRegistry(
    (
        ParameterDefinition(
            path="compute.batch_size",
            label="Batch size",
            category="Compute",
            python_types=(int,),
            cli_argument="--batch-size",
            affects_placement=True,
            supported_by=frozenset(
                {"llama-bench", "llama-server", "llama-fit-params"}
            ),
            minimum=1,
        ),
        ParameterDefinition(
            path="compute.ubatch_size",
            label="Physical batch size",
            category="Compute",
            python_types=(int,),
            cli_argument="--ubatch-size",
            affects_placement=True,
            supported_by=frozenset(
                {"llama-bench", "llama-server", "llama-fit-params"}
            ),
            minimum=1,
        ),
        ParameterDefinition(
            path="compute.flash_attn",
            label="Flash attention",
            category="Compute",
            python_types=(str,),
            cli_argument="--flash-attn",
            affects_placement=True,
            supported_by=frozenset(
                {"llama-bench", "llama-server", "llama-fit-params"}
            ),
            string_choices=frozenset({"on", "off", "auto"}),
        ),
        ParameterDefinition(
            path="compute.threads",
            label="Threads",
            category="Compute",
            python_types=(int,),
            cli_argument="--threads",
            affects_placement=False,
            supported_by=frozenset({"llama-bench", "llama-server"}),
            minimum=1,
        ),
        ParameterDefinition(
            path="context.size",
            label="Context size",
            category="Context",
            python_types=(int,),
            cli_argument="--ctx-size",
            affects_placement=True,
            supported_by=frozenset({"llama-server", "llama-fit-params"}),
            minimum=1,
        ),
        ParameterDefinition(
            path="context.cache_type_k",
            label="KV cache K type",
            category="Context",
            python_types=(str,),
            cli_argument="--cache-type-k",
            affects_placement=True,
            supported_by=frozenset(
                {"llama-bench", "llama-server", "llama-fit-params"}
            ),
            string_choices=_CACHE_TYPES,
        ),
        ParameterDefinition(
            path="context.cache_type_v",
            label="KV cache V type",
            category="Context",
            python_types=(str,),
            cli_argument="--cache-type-v",
            affects_placement=True,
            supported_by=frozenset(
                {"llama-bench", "llama-server", "llama-fit-params"}
            ),
            string_choices=_CACHE_TYPES,
        ),
        ParameterDefinition(
            path="placement.fit.target_mib",
            label="Fit target",
            category="Placement",
            python_types=(int,),
            cli_argument="--fit-target",
            affects_placement=True,
            supported_by=frozenset({"llama-server", "llama-fit-params"}),
            minimum=0,
        ),
        ParameterDefinition(
            path="placement.constraints.n_gpu_layers",
            label="GPU layers",
            category="Placement",
            python_types=(int, str),
            cli_argument="--n-gpu-layers",
            affects_placement=True,
            supported_by=frozenset(
                {"llama-bench", "llama-server", "llama-fit-params"}
            ),
            minimum=0,
            string_choices=frozenset({"auto", "all"}),
        ),
        ParameterDefinition(
            path="speculative.enabled",
            label="Speculative decoding",
            category="Speculative decoding",
            python_types=(bool,),
            cli_argument=None,
            affects_placement=False,
            supported_by=frozenset({"llama-server"}),
        ),
        ParameterDefinition(
            path="speculative.type",
            label="Speculative type",
            category="Speculative decoding",
            python_types=(str, type(None)),
            cli_argument="--spec-type",
            affects_placement=False,
            supported_by=frozenset({"llama-server"}),
        ),
        ParameterDefinition(
            path="speculative.draft_n_max",
            label="Maximum draft tokens",
            category="Speculative decoding",
            python_types=(int,),
            cli_argument="--spec-draft-n-max",
            affects_placement=False,
            supported_by=frozenset({"llama-server"}),
            minimum=1,
        ),
    )
)
