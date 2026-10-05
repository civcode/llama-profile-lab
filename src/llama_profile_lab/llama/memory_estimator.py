"""Structured per-device memory-estimator helper adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from llama_profile_lab.domain import (
    Candidate,
    MemoryEstimateIdentity,
    MemoryEstimateOutput,
)
from llama_profile_lab.llama.capabilities import CapabilitySet


class MemoryEstimatorConfigurationError(ValueError):
    """Raised when a Candidate cannot be represented by the helper contract."""


class MemoryEstimatorParseError(ValueError):
    """Raised when structured helper output is invalid."""


@dataclass(frozen=True, slots=True)
class MemoryEstimatorInvocation:
    """Complete deterministic helper request."""

    identity: MemoryEstimateIdentity
    argv: tuple[str, ...]


class MemoryEstimatorAdapter:
    """Translate a Candidate into the version-1 memory-helper contract."""

    REQUIRED_OPTIONS = frozenset(
        {
            "--json",
            "--model",
            "--ctx-size",
            "--batch-size",
            "--ubatch-size",
            "--cache-type-k",
            "--cache-type-v",
            "--gpu-layers",
            "--split-mode",
            "--main-gpu",
        }
    )

    def build_invocation(
        self,
        *,
        binary_path: Path,
        capabilities: CapabilitySet,
        helper_sha256: str,
        host_id: str,
        model_path: Path,
        candidate: Candidate,
        selected_devices: tuple[str, ...] | None = None,
    ) -> MemoryEstimatorInvocation:
        """Build argv and the cache identity for one estimate."""
        if capabilities.kind != "llama-memory-estimator":
            raise MemoryEstimatorConfigurationError(
                f"selected binary is {capabilities.kind}, "
                "not llama-memory-estimator"
            )
        missing = sorted(
            option
            for option in self.REQUIRED_OPTIONS
            if not capabilities.supports(option)
        )
        if missing:
            raise MemoryEstimatorConfigurationError(
                "memory estimator does not advertise required options: "
                + ", ".join(missing)
            )

        constraints = candidate.placement.constraints
        effective_devices = (
            selected_devices
            if selected_devices is not None
            else (() if constraints.devices == "auto" else constraints.devices)
        )
        if len(effective_devices) != len(set(effective_devices)):
            raise MemoryEstimatorConfigurationError(
                "selected devices must be unique and ordered"
            )
        tensor_split = constraints.tensor_split
        if (
            tensor_split is not None
            and effective_devices
            and len(tensor_split) != len(effective_devices)
        ):
            raise MemoryEstimatorConfigurationError(
                "tensor split must contain one value per selected device"
            )

        n_gpu_layers = constraints.n_gpu_layers
        identity = MemoryEstimateIdentity(
            host_id=host_id,
            candidate_hash=candidate.content_hash(),
            model_artifact_id=candidate.model.target_model_id,
            helper_sha256=helper_sha256,
            selected_devices=effective_devices,
            n_gpu_layers=n_gpu_layers,
            split_mode=constraints.split_mode,
            main_gpu=constraints.main_gpu,
            tensor_split=tensor_split,
            override_tensor=constraints.override_tensor,
        )

        argv = [
            str(binary_path),
            "--json",
            "--model",
            str(model_path),
            "--ctx-size",
            str(candidate.context.size),
            "--batch-size",
            str(candidate.compute.batch_size),
            "--ubatch-size",
            str(candidate.compute.ubatch_size),
            "--cache-type-k",
            candidate.context.cache_type_k,
            "--cache-type-v",
            candidate.context.cache_type_v,
            "--gpu-layers",
            _gpu_layers_arg(n_gpu_layers),
            "--split-mode",
            constraints.split_mode,
            "--main-gpu",
            str(constraints.main_gpu),
        ]
        if effective_devices:
            _require(capabilities, "--device")
            argv.extend(("--device", ",".join(effective_devices)))
        if tensor_split is not None:
            _require(capabilities, "--tensor-split")
            argv.extend(
                (
                    "--tensor-split",
                    ",".join(_format_float(value) for value in tensor_split),
                )
            )
        for override in constraints.override_tensor:
            _require(capabilities, "--override-tensor")
            argv.extend(("--override-tensor", override))

        if candidate.compute.flash_attn != "auto":
            _require(capabilities, "--flash-attn")
            argv.extend(("--flash-attn", candidate.compute.flash_attn))
        if candidate.compute.load_mode != "auto":
            _require(capabilities, "--load-mode")
            argv.extend(("--load-mode", candidate.compute.load_mode))
        if candidate.compute.lazy_mode != "auto":
            _require(capabilities, "--lazy-mode")
            argv.extend(("--lazy-mode", candidate.compute.lazy_mode))
        if not candidate.context.kv_offload:
            _require(capabilities, "--no-kv-offload")
            argv.append("--no-kv-offload")
        if candidate.compute.no_op_offload:
            _require(capabilities, "--no-op-offload")
            argv.append("--no-op-offload")
        if candidate.compute.no_host:
            _require(capabilities, "--no-host")
            argv.append("--no-host")
        if not candidate.compute.repack:
            _require(capabilities, "--no-repack")
            argv.append("--no-repack")

        return MemoryEstimatorInvocation(identity=identity, argv=tuple(argv))

    def parse_output(self, stdout: str) -> MemoryEstimateOutput:
        """Parse the helper's strict versioned JSON stdout contract."""
        try:
            raw = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise MemoryEstimatorParseError(
                f"memory estimator did not emit valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(raw, dict):
            raise MemoryEstimatorParseError(
                "memory estimator JSON document must be an object"
            )
        try:
            return MemoryEstimateOutput.model_validate(raw)
        except ValidationError as exc:
            raise MemoryEstimatorParseError(
                f"invalid memory estimator JSON: {exc}"
            ) from exc


def _gpu_layers_arg(value: int | str | None) -> str:
    if value is None:
        return "auto"
    return str(value)


def _format_float(value: float) -> str:
    return format(value, ".17g")


def _require(capabilities: CapabilitySet, option: str) -> None:
    if not capabilities.supports(option):
        raise MemoryEstimatorConfigurationError(
            f"memory estimator does not advertise required option {option}"
        )
