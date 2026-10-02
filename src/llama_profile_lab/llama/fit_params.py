"""llama-fit-params adapter and fitted placement parser."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from llama_profile_lab.domain import Candidate, ResolvedPlacement
from llama_profile_lab.domain.candidate import ExtraArgument
from llama_profile_lab.llama.capabilities import CapabilitySet


class LlamaFitParamsConfigurationError(ValueError):
    """Raised when a Candidate cannot be safely fitted."""


class LlamaFitParamsParseError(ValueError):
    """Raised when fitted CLI arguments cannot be normalized."""


@dataclass(frozen=True, slots=True)
class LlamaFitParamsResult:
    """Parsed concrete placement plus the exact fitted argv fragment."""

    placement: ResolvedPlacement
    fitted_argv: tuple[str, ...]


class LlamaFitParamsAdapter:
    """Translate one Candidate into a full-production-context fit invocation."""

    def build_argv(
        self,
        *,
        binary_path: Path,
        capabilities: CapabilitySet,
        model_path: Path,
        candidate: Candidate,
    ) -> tuple[str, ...]:
        """Build argv for fitting exactly the Candidate's production context."""
        if capabilities.kind != "llama-fit-params":
            raise LlamaFitParamsConfigurationError(
                f"selected binary is {capabilities.kind}, not llama-fit-params"
            )
        if candidate.placement.mode != "fit" or candidate.placement.fit is None:
            raise LlamaFitParamsConfigurationError(
                "per-candidate placement resolution requires Candidate placement.mode=fit"
            )

        constraints = candidate.placement.constraints
        if constraints.n_gpu_layers not in {None, "auto"}:
            raise LlamaFitParamsConfigurationError(
                "automatic fit cannot safely combine with a fixed n_gpu_layers; "
                "use an experiment fixed placement instead"
            )
        if constraints.n_cpu_moe != 0:
            raise LlamaFitParamsConfigurationError(
                "automatic fit cannot safely combine with fixed n_cpu_moe"
            )
        if constraints.tensor_split is not None:
            raise LlamaFitParamsConfigurationError(
                "automatic fit cannot safely combine with fixed tensor_split"
            )
        if constraints.override_tensor:
            raise LlamaFitParamsConfigurationError(
                "automatic fit cannot safely combine with fixed override_tensor"
            )

        argv: list[str] = [str(binary_path)]
        _append(argv, capabilities, "--model", ("-m",), str(model_path))
        _append(
            argv,
            capabilities,
            "--ctx-size",
            ("-c",),
            str(candidate.context.size),
        )
        _append(
            argv,
            capabilities,
            "--batch-size",
            ("-b",),
            str(candidate.compute.batch_size),
        )
        _append(
            argv,
            capabilities,
            "--ubatch-size",
            ("-ub",),
            str(candidate.compute.ubatch_size),
        )
        _append(
            argv,
            capabilities,
            "--cache-type-k",
            ("-ctk",),
            candidate.context.cache_type_k,
        )
        _append(
            argv,
            capabilities,
            "--cache-type-v",
            ("-ctv",),
            candidate.context.cache_type_v,
        )

        if candidate.compute.flash_attn != "auto":
            _append(
                argv,
                capabilities,
                "--flash-attn",
                ("-fa",),
                candidate.compute.flash_attn,
            )
        if candidate.compute.load_mode != "auto":
            _append(
                argv,
                capabilities,
                "--load-mode",
                (),
                candidate.compute.load_mode,
            )
        if candidate.compute.lazy_mode != "auto":
            _append(
                argv,
                capabilities,
                "--lazy-mode",
                (),
                candidate.compute.lazy_mode,
            )

        fit = candidate.placement.fit
        _append(
            argv,
            capabilities,
            "--fit-target",
            ("-fitt",),
            str(fit.target_mib),
        )
        _append(
            argv,
            capabilities,
            "--fit-ctx",
            ("-fitc",),
            str(fit.min_context),
        )

        if constraints.split_mode != "layer":
            _append(
                argv,
                capabilities,
                "--split-mode",
                ("-sm",),
                constraints.split_mode,
            )
        if constraints.main_gpu:
            _append(
                argv,
                capabilities,
                "--main-gpu",
                ("-mg",),
                str(constraints.main_gpu),
            )
        if constraints.devices != "auto":
            _append(
                argv,
                capabilities,
                "--device",
                ("-dev",),
                ",".join(constraints.devices),
            )

        _append_boolean_switch(
            argv,
            capabilities,
            enabled=candidate.context.kv_offload,
            positive=("--kv-offload", "-kvo"),
            negative=("--no-kv-offload", "-nkvo"),
            label="KV offload",
        )
        _append_boolean_switch(
            argv,
            capabilities,
            enabled=not candidate.compute.no_op_offload,
            positive=("--op-offload",),
            negative=("--no-op-offload",),
            label="operation offload",
        )
        if candidate.compute.no_host:
            _append_flag(argv, capabilities, "--no-host", ())
        _append_boolean_switch(
            argv,
            capabilities,
            enabled=candidate.compute.repack,
            positive=("--repack",),
            negative=("--no-repack", "-nr"),
            label="weight repacking",
        )

        for extra in candidate.extra_args:
            _append_extra(argv, capabilities, extra)

        return tuple(argv)

    def parse_output(
        self,
        stdout: str,
        *,
        candidate: Candidate,
    ) -> LlamaFitParamsResult:
        """Parse the final fitted CLI argument line from stdout."""
        tokens = _find_fitted_tokens(stdout)
        parsed = _parse_fitted_tokens(tokens)
        context_size = parsed.context_size
        if context_size != candidate.context.size:
            raise LlamaFitParamsParseError(
                "llama-fit-params changed production context despite explicit ctx-size: "
                f"expected {candidate.context.size}, got {context_size}"
            )

        constraints = candidate.placement.constraints
        placement = ResolvedPlacement(
            production_context_size=context_size,
            n_gpu_layers=parsed.n_gpu_layers,
            n_cpu_moe=constraints.n_cpu_moe,
            split_mode=constraints.split_mode,
            main_gpu=constraints.main_gpu,
            devices=constraints.devices,
            tensor_split=parsed.tensor_split,
            override_tensor=parsed.override_tensor,
        )
        return LlamaFitParamsResult(
            placement=placement,
            fitted_argv=tokens,
        )


@dataclass(frozen=True, slots=True)
class _ParsedFitTokens:
    context_size: int
    n_gpu_layers: int
    tensor_split: tuple[float, ...] | None
    override_tensor: tuple[str, ...]


def _find_fitted_tokens(stdout: str) -> tuple[str, ...]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        try:
            tokens = tuple(shlex.split(stripped))
        except ValueError:
            continue
        if any(token in {"-c", "--ctx-size"} for token in tokens) and any(
            token in {"-ngl", "--gpu-layers", "--n-gpu-layers"} for token in tokens
        ):
            return tokens
    raise LlamaFitParamsParseError(
        "could not find fitted context/GPU-layer argument line in stdout"
    )


def _parse_fitted_tokens(tokens: tuple[str, ...]) -> _ParsedFitTokens:
    context_size: int | None = None
    n_gpu_layers: int | None = None
    tensor_split: tuple[float, ...] | None = None
    override_tensor_values: list[str] = []

    index = 0
    while index < len(tokens):
        option = tokens[index]
        if option in {
            "-c",
            "--ctx-size",
            "-ngl",
            "--gpu-layers",
            "--n-gpu-layers",
            "-ts",
            "--tensor-split",
            "-ot",
            "--override-tensor",
        }:
            if index + 1 >= len(tokens):
                raise LlamaFitParamsParseError(
                    f"fitted option {option} is missing its value"
                )
            value = tokens[index + 1]
            index += 2
        else:
            raise LlamaFitParamsParseError(
                f"unexpected fitted placement option: {option}"
            )

        if option in {"-c", "--ctx-size"}:
            context_size = _positive_int(value, option)
        elif option in {"-ngl", "--gpu-layers", "--n-gpu-layers"}:
            n_gpu_layers = _non_negative_int(value, option)
        elif option in {"-ts", "--tensor-split"}:
            tensor_split = _tensor_split(value)
        elif option in {"-ot", "--override-tensor"}:
            # Keep the complete comma-separated override expression intact.
            # Regex patterns may themselves contain punctuation.
            override_tensor_values.append(value)

    if context_size is None or n_gpu_layers is None:
        raise LlamaFitParamsParseError(
            "fitted output must contain context size and n_gpu_layers"
        )
    return _ParsedFitTokens(
        context_size=context_size,
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        override_tensor=tuple(override_tensor_values),
    )


def _positive_int(value: str, option: str) -> int:
    parsed = _non_negative_int(value, option)
    if parsed == 0:
        raise LlamaFitParamsParseError(f"{option} must be positive")
    return parsed


def _non_negative_int(value: str, option: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise LlamaFitParamsParseError(
            f"{option} is not an integer: {value!r}"
        ) from exc
    if parsed < 0:
        raise LlamaFitParamsParseError(f"{option} must be non-negative")
    return parsed


def _tensor_split(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise LlamaFitParamsParseError(
            f"invalid tensor split: {value!r}"
        ) from exc
    if not values or any(item < 0 for item in values):
        raise LlamaFitParamsParseError(f"invalid tensor split: {value!r}")
    return values


def _append(
    argv: list[str],
    capabilities: CapabilitySet,
    preferred: str,
    fallbacks: tuple[str, ...],
    value: str,
) -> None:
    option = _select_option(capabilities, preferred, fallbacks)
    argv.extend((option, value))


def _append_flag(
    argv: list[str],
    capabilities: CapabilitySet,
    preferred: str,
    fallbacks: tuple[str, ...],
) -> None:
    argv.append(_select_option(capabilities, preferred, fallbacks))


def _append_boolean_switch(
    argv: list[str],
    capabilities: CapabilitySet,
    *,
    enabled: bool,
    positive: tuple[str, ...],
    negative: tuple[str, ...],
    label: str,
) -> None:
    choices = positive if enabled else negative
    if not choices:
        if enabled:
            raise LlamaFitParamsConfigurationError(
                f"fit binary cannot explicitly enable {label}"
            )
        return
    preferred, *fallbacks = choices
    try:
        option = _select_option(
            capabilities,
            preferred,
            tuple(fallbacks),
        )
    except LlamaFitParamsConfigurationError as exc:
        raise LlamaFitParamsConfigurationError(
            f"fit binary cannot represent {label}={enabled}"
        ) from exc
    argv.append(option)


_PLACEMENT_CONTROL_OPTIONS = frozenset(
    {
        "--fit",
        "-fit",
        "--fit-target",
        "-fitt",
        "--fit-ctx",
        "-fitc",
        "--ctx-size",
        "-c",
        "--n-gpu-layers",
        "--gpu-layers",
        "-ngl",
        "--tensor-split",
        "-ts",
        "--override-tensor",
        "-ot",
    }
)


def _append_extra(
    argv: list[str],
    capabilities: CapabilitySet,
    extra: ExtraArgument,
) -> None:
    if extra.name in _PLACEMENT_CONTROL_OPTIONS:
        raise LlamaFitParamsConfigurationError(
            f"Candidate extra arg {extra.name} conflicts with placement resolution"
        )
    if not capabilities.supports(extra.name):
        raise LlamaFitParamsConfigurationError(
            f"fit binary does not advertise Candidate extra arg {extra.name}"
        )
    argv.append(extra.name)
    if extra.value is None:
        return
    if isinstance(extra.value, tuple):
        if any(item is None for item in extra.value):
            raise LlamaFitParamsConfigurationError(
                f"Candidate extra arg {extra.name} cannot contain null tuple values"
            )
        argv.append(
            ",".join(str(item) for item in extra.value if item is not None)
        )
    elif isinstance(extra.value, bool):
        argv.append("1" if extra.value else "0")
    else:
        argv.append(str(extra.value))


def _select_option(
    capabilities: CapabilitySet,
    preferred: str,
    fallbacks: tuple[str, ...],
) -> str:
    if capabilities.supports(preferred):
        return preferred
    for fallback in fallbacks:
        if capabilities.supports(fallback):
            return fallback
    raise LlamaFitParamsConfigurationError(
        f"fit binary does not support required option {preferred}"
    )
