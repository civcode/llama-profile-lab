"""llama-server argv construction for finalist validation."""

from __future__ import annotations

from pathlib import Path

from llama_profile_lab.domain import Candidate, ResolvedPlacement
from llama_profile_lab.domain.candidate import ExtraArgument
from llama_profile_lab.llama.capabilities import CapabilitySet


class LlamaServerConfigurationError(ValueError):
    """Raised when a Candidate cannot be represented by the selected server."""


class LlamaServerAdapter:
    """Translate one resolved Candidate into a reproducible llama-server argv."""

    def build_argv(
        self,
        *,
        binary_path: Path,
        capabilities: CapabilitySet,
        model_path: Path,
        candidate: Candidate,
        placement: ResolvedPlacement,
        host: str,
        port: int,
        draft_model_path: Path | None = None,
        model_alias: str | None = None,
    ) -> tuple[str, ...]:
        if capabilities.kind != "llama-server":
            raise LlamaServerConfigurationError(
                f"selected binary is {capabilities.kind}, not llama-server"
            )
        if not host:
            raise LlamaServerConfigurationError("server host must not be empty")
        if not 1 <= port <= 65535:
            raise LlamaServerConfigurationError("server port must be between 1 and 65535")
        if placement.production_context_size != candidate.context.size:
            raise LlamaServerConfigurationError(
                "resolved placement production context does not match Candidate"
            )

        argv: list[str] = [str(binary_path)]
        _append(argv, capabilities, "--model", ("-m",), str(model_path))
        _append(argv, capabilities, "--host", (), host)
        _append(argv, capabilities, "--port", (), str(port))
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
        _append(
            argv,
            capabilities,
            "--parallel",
            ("-np",),
            str(candidate.server.parallel),
        )

        if candidate.compute.threads is not None:
            _append(
                argv,
                capabilities,
                "--threads",
                ("-t",),
                str(candidate.compute.threads),
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
                ("-lm",),
                candidate.compute.load_mode,
            )
        if candidate.compute.lazy_mode != "auto":
            _append(
                argv,
                capabilities,
                "--lazy-mode",
                ("-lzm",),
                candidate.compute.lazy_mode,
            )

        if placement.n_gpu_layers is not None:
            _append(
                argv,
                capabilities,
                "--n-gpu-layers",
                ("-ngl", "--gpu-layers"),
                str(placement.n_gpu_layers),
            )
        if placement.n_cpu_moe:
            _append(
                argv,
                capabilities,
                "--n-cpu-moe",
                ("-ncmoe",),
                str(placement.n_cpu_moe),
            )
        if placement.split_mode != "layer":
            _append(
                argv,
                capabilities,
                "--split-mode",
                ("-sm",),
                placement.split_mode,
            )
        if placement.main_gpu:
            _append(
                argv,
                capabilities,
                "--main-gpu",
                ("-mg",),
                str(placement.main_gpu),
            )
        if placement.devices != "auto":
            _append(
                argv,
                capabilities,
                "--device",
                ("-dev",),
                ",".join(placement.devices),
            )
        if placement.tensor_split is not None:
            _append(
                argv,
                capabilities,
                "--tensor-split",
                ("-ts",),
                ",".join(str(value) for value in placement.tensor_split),
            )
        for override in placement.override_tensor:
            _append(
                argv,
                capabilities,
                "--override-tensor",
                ("-ot",),
                override,
            )

        _append_boolean_pair(
            argv,
            capabilities,
            true_option="--kv-offload",
            true_fallbacks=("-kvo",),
            false_option="--no-kv-offload",
            false_fallbacks=("-nkvo",),
            value=candidate.context.kv_offload,
            default_value=True,
        )
        _append_boolean_pair(
            argv,
            capabilities,
            true_option="--kv-unified",
            true_fallbacks=("-kvu",),
            false_option="--no-kv-unified",
            false_fallbacks=("-no-kvu",),
            value=candidate.context.kv_unified,
            default_value=True,
        )
        _append_boolean_pair(
            argv,
            capabilities,
            true_option="--op-offload",
            true_fallbacks=(),
            false_option="--no-op-offload",
            false_fallbacks=(),
            value=not candidate.compute.no_op_offload,
            default_value=True,
        )
        _append_boolean_pair(
            argv,
            capabilities,
            true_option="--repack",
            true_fallbacks=(),
            false_option="--no-repack",
            false_fallbacks=("-nr",),
            value=candidate.compute.repack,
            default_value=True,
        )
        if candidate.compute.no_host:
            _append_flag(argv, capabilities, "--no-host", ())

        if model_alias is not None:
            _append(argv, capabilities, "--alias", ("-a",), model_alias)

        if candidate.speculative.enabled:
            if candidate.speculative.type is None or candidate.speculative.draft_n_max is None:
                raise LlamaServerConfigurationError(
                    "enabled speculative decoding is incomplete"
                )
            _append(
                argv,
                capabilities,
                "--spec-type",
                (),
                candidate.speculative.type,
            )
            _append(
                argv,
                capabilities,
                "--spec-draft-n-max",
                (),
                str(candidate.speculative.draft_n_max),
            )
            if candidate.model.draft_model_id is not None and draft_model_path is None:
                raise LlamaServerConfigurationError(
                    "Candidate references a draft model but no draft model path was supplied"
                )
            if draft_model_path is not None:
                _append(
                    argv,
                    capabilities,
                    "--spec-draft-model",
                    ("--model-draft", "-md"),
                    str(draft_model_path),
                )
        elif draft_model_path is not None:
            raise LlamaServerConfigurationError(
                "draft model path supplied for non-speculative Candidate"
            )

        for extra in candidate.extra_args:
            _append_extra(argv, capabilities, extra)

        return tuple(argv)


_CONTROLLED_OPTIONS = frozenset(
    {
        "--model",
        "-m",
        "--host",
        "--port",
        "--ctx-size",
        "-c",
        "--batch-size",
        "-b",
        "--ubatch-size",
        "-ub",
        "--cache-type-k",
        "-ctk",
        "--cache-type-v",
        "-ctv",
        "--parallel",
        "-np",
        "--n-gpu-layers",
        "--gpu-layers",
        "-ngl",
        "--n-cpu-moe",
        "-ncmoe",
        "--split-mode",
        "-sm",
        "--main-gpu",
        "-mg",
        "--device",
        "-dev",
        "--tensor-split",
        "-ts",
        "--override-tensor",
        "-ot",
        "--kv-offload",
        "--no-kv-offload",
        "--kv-unified",
        "--no-kv-unified",
        "--op-offload",
        "--no-op-offload",
        "--repack",
        "--no-repack",
        "--spec-type",
        "--spec-draft-n-max",
        "--spec-draft-model",
        "--model-draft",
        "-md",
    }
)


def _append(
    argv: list[str],
    capabilities: CapabilitySet,
    preferred: str,
    fallbacks: tuple[str, ...],
    value: str,
) -> None:
    argv.extend((_select_option(capabilities, preferred, fallbacks), value))


def _append_flag(
    argv: list[str],
    capabilities: CapabilitySet,
    preferred: str,
    fallbacks: tuple[str, ...],
) -> None:
    argv.append(_select_option(capabilities, preferred, fallbacks))


def _append_boolean_pair(
    argv: list[str],
    capabilities: CapabilitySet,
    *,
    true_option: str,
    true_fallbacks: tuple[str, ...],
    false_option: str,
    false_fallbacks: tuple[str, ...],
    value: bool,
    default_value: bool,
) -> None:
    preferred = true_option if value else false_option
    fallbacks = true_fallbacks if value else false_fallbacks
    if capabilities.supports(preferred):
        argv.append(preferred)
        return
    for fallback in fallbacks:
        if capabilities.supports(fallback):
            argv.append(fallback)
            return
    if value == default_value:
        return
    raise LlamaServerConfigurationError(
        f"server binary cannot represent required boolean option {preferred}"
    )


def _append_extra(
    argv: list[str],
    capabilities: CapabilitySet,
    extra: ExtraArgument,
) -> None:
    if extra.name in _CONTROLLED_OPTIONS:
        raise LlamaServerConfigurationError(
            f"Candidate extra arg {extra.name} conflicts with server-managed option"
        )
    if not capabilities.supports(extra.name):
        raise LlamaServerConfigurationError(
            f"server does not advertise Candidate extra arg {extra.name}"
        )
    argv.append(extra.name)
    value = extra.value
    if value is None:
        return
    if isinstance(value, tuple):
        if any(item is None for item in value):
            raise LlamaServerConfigurationError(
                f"Candidate extra arg {extra.name} cannot contain null tuple values"
            )
        argv.append(",".join(_scalar_arg(item) for item in value if item is not None))
        return
    argv.append(_scalar_arg(value))


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
    raise LlamaServerConfigurationError(
        f"server binary does not support required option {preferred}"
    )


def _scalar_arg(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)
