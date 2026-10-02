"""Placement caching and llama-fit-params execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Literal

from llama_profile_lab.db import PlacementRepository
from llama_profile_lab.db.records import BinaryRecord, ResolvedPlacementRecord
from llama_profile_lab.domain import Candidate, ResolvedPlacement, sha256_json
from llama_profile_lab.execution.process import ProcessRunner, ProcessRunnerError
from llama_profile_lab.llama import CapabilitySet, sha256_file
from llama_profile_lab.llama.fit_params import (
    LlamaFitParamsAdapter,
    LlamaFitParamsConfigurationError,
    LlamaFitParamsParseError,
)


class PlacementConfigurationError(RuntimeError):
    """Raised when placement policy or binary identity is invalid."""


PlacementFailureKind = Literal[
    "fit_failed",
    "timeout",
    "parser_failed",
    "interrupted",
    "cancelled",
]


class PlacementResolutionFailure(RuntimeError):
    """Raised after a failed fit attempt has already been persisted."""

    def __init__(self, message: str, *, kind: PlacementFailureKind) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class PlacementResolution:
    """Successful placement resolution plus cache behavior."""

    record: ResolvedPlacementRecord
    cache_hit: bool


class PlacementResolver:
    """Resolve and cache full-context placement for one Candidate."""

    def __init__(
        self,
        repository: PlacementRepository,
        *,
        process_runner: ProcessRunner,
        adapter: LlamaFitParamsAdapter | None = None,
    ) -> None:
        self.repository = repository
        self.process_runner = process_runner
        self.adapter = adapter or LlamaFitParamsAdapter()

    def resolve_per_candidate(
        self,
        *,
        candidate_id: str,
        candidate: Candidate,
        host_id: str,
        hardware_fingerprint: str,
        fit_binary: BinaryRecord,
        model_path: Path,
        timeout_seconds: float | None = None,
        cancel_event: Event | None = None,
    ) -> PlacementResolution:
        """Resolve one Candidate at production context or return a cache hit."""
        _verify_binary(fit_binary, expected_kind="llama-fit-params")

        resolved_model_path = model_path.expanduser().resolve()
        if not resolved_model_path.is_file():
            raise PlacementConfigurationError(
                f"model does not exist: {resolved_model_path}"
            )

        request = placement_request(
            candidate_id=candidate_id,
            candidate=candidate,
            hardware_fingerprint=hardware_fingerprint,
            fit_binary=fit_binary,
            model_path=resolved_model_path,
        )
        placement_hash = sha256_json(request)
        cached = self.repository.find_by_hash(placement_hash)
        if cached is not None:
            return PlacementResolution(record=cached, cache_hit=True)

        try:
            argv = self.adapter.build_argv(
                binary_path=Path(fit_binary.path),
                capabilities=_capabilities(fit_binary),
                model_path=resolved_model_path,
                candidate=candidate,
            )
        except LlamaFitParamsConfigurationError as exc:
            attempt_id = self.repository.create_attempt(
                placement_hash=placement_hash,
                candidate_id=candidate_id,
                host_id=host_id,
                binary_id=fit_binary.id,
                model_path=str(resolved_model_path),
                argv=(),
            )
            self.repository.finish_attempt(
                attempt_id,
                status="fit_failed",
                duration_ns=0,
                exit_code=None,
                stderr=str(exc),
            )
            raise PlacementResolutionFailure(
                str(exc),
                kind="fit_failed",
            ) from exc

        attempt_id = self.repository.create_attempt(
            placement_hash=placement_hash,
            candidate_id=candidate_id,
            host_id=host_id,
            binary_id=fit_binary.id,
            model_path=str(resolved_model_path),
            argv=argv,
        )
        try:
            process = self.process_runner.run(
                argv,
                timeout_seconds=timeout_seconds,
                cancel_event=cancel_event,
            )
        except ProcessRunnerError as exc:
            self.repository.finish_attempt(
                attempt_id,
                status="fit_failed",
                duration_ns=0,
                exit_code=None,
                stderr=str(exc),
            )
            raise PlacementResolutionFailure(
                str(exc),
                kind="fit_failed",
            ) from exc

        terminal_kind = _fit_process_failure_kind(process)
        if terminal_kind is not None:
            self.repository.finish_attempt(
                attempt_id,
                status=terminal_kind,
                duration_ns=process.duration_ns,
                exit_code=process.exit_code,
                stdout=process.stdout,
                stderr=process.stderr,
                finished_at=process.finished_at,
            )
            raise PlacementResolutionFailure(
                f"llama-fit-params ended with {terminal_kind}",
                kind=terminal_kind,
            )

        try:
            result = self.adapter.parse_output(process.stdout, candidate=candidate)
        except LlamaFitParamsParseError as exc:
            self.repository.finish_attempt(
                attempt_id,
                status="parser_failed",
                duration_ns=process.duration_ns,
                exit_code=process.exit_code,
                stdout=process.stdout,
                stderr=_append_error(process.stderr, str(exc)),
                finished_at=process.finished_at,
            )
            raise PlacementResolutionFailure(
                str(exc),
                kind="parser_failed",
            ) from exc

        raw_result = {
            "placement": result.placement.model_dump(mode="json"),
            "fitted_argv": list(result.fitted_argv),
        }
        self.repository.finish_attempt(
            attempt_id,
            status="completed",
            duration_ns=process.duration_ns,
            exit_code=process.exit_code,
            stdout=process.stdout,
            stderr=process.stderr,
            raw_result=raw_result,
            finished_at=process.finished_at,
        )
        placement_id = self.repository.put_resolved(
            placement_hash=placement_hash,
            candidate_id=candidate_id,
            host_id=host_id,
            binary_id=fit_binary.id,
            fit_attempt_id=attempt_id,
            placement=result.placement,
            request=request,
            argv=argv,
            stdout=process.stdout,
            stderr=process.stderr,
            exit_code=process.exit_code,
            raw_result=raw_result,
        )
        record = self.repository.get(placement_id)
        if record is None:
            raise RuntimeError("resolved placement was not readable after insertion")
        return PlacementResolution(record=record, cache_hit=False)


def placement_request(
    *,
    candidate_id: str,
    candidate: Candidate,
    hardware_fingerprint: str,
    fit_binary: BinaryRecord,
    model_path: Path,
) -> dict[str, Any]:
    """Build the deterministic cache-key document for one fit request."""
    stat = model_path.stat()
    return {
        "schema": "llama-placement-request",
        "version": 1,
        "candidate_id": candidate_id,
        "target_model_id": candidate.model.target_model_id,
        "production_context_size": candidate.context.size,
        "host_hardware_fingerprint": hardware_fingerprint,
        "fit_binary_sha256": fit_binary.sha256,
        "model_artifact": {
            "path": str(model_path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "fit_input": {
            "context": {
                "cache_type_k": candidate.context.cache_type_k,
                "cache_type_v": candidate.context.cache_type_v,
                "kv_offload": candidate.context.kv_offload,
            },
            "compute": {
                "batch_size": candidate.compute.batch_size,
                "ubatch_size": candidate.compute.ubatch_size,
                "flash_attn": candidate.compute.flash_attn,
                "load_mode": candidate.compute.load_mode,
                "lazy_mode": candidate.compute.lazy_mode,
                "repack": candidate.compute.repack,
                "no_host": candidate.compute.no_host,
                "no_op_offload": candidate.compute.no_op_offload,
            },
            "fit": (
                candidate.placement.fit.model_dump(mode="json")
                if candidate.placement.fit is not None
                else None
            ),
            "constraints": candidate.placement.constraints.model_dump(mode="json"),
            "extra_args": {
                extra.name: extra.value
                for extra in candidate.extra_args
            },
        },
    }


def resolved_placement_from_record(
    record: ResolvedPlacementRecord,
) -> ResolvedPlacement:
    """Convert persistence DTO into the immutable placement domain object."""
    if isinstance(record.devices, str):
        if record.devices != "auto":
            raise PlacementConfigurationError(
                f"invalid persisted device setting: {record.devices!r}"
            )
        devices = "auto"
    else:
        devices = record.devices

    return ResolvedPlacement(
        production_context_size=record.production_context_size,
        n_gpu_layers=record.n_gpu_layers,
        n_cpu_moe=record.n_cpu_moe,
        split_mode=record.split_mode,
        main_gpu=record.main_gpu,
        devices=devices,
        tensor_split=record.tensor_split,
        override_tensor=record.override_tensor,
    )


def validate_fixed_placement(
    record: ResolvedPlacementRecord,
    *,
    candidate: Candidate,
    host_id: str,
) -> None:
    """Reject a fixed placement that does not represent this host/context."""
    if record.host_id != host_id:
        raise PlacementConfigurationError(
            "fixed placement belongs to a different host hardware identity"
        )
    if record.production_context_size != candidate.context.size:
        raise PlacementConfigurationError(
            "fixed placement production context does not match Candidate context"
        )


def _verify_binary(binary: BinaryRecord, *, expected_kind: str) -> None:
    if binary.kind != expected_kind:
        raise PlacementConfigurationError(
            f"binary {binary.id} is {binary.kind}, not {expected_kind}"
        )
    path = Path(binary.path)
    if not path.is_file():
        raise PlacementConfigurationError(
            f"registered {expected_kind} path does not exist: {path}"
        )
    if sha256_file(path) != binary.sha256:
        raise PlacementConfigurationError(
            f"registered {expected_kind} binary changed on disk; "
            "re-run binary inspect"
        )


def _capabilities(binary: BinaryRecord) -> CapabilitySet:
    return CapabilitySet.from_mapping(
        dict(binary.capabilities),
        fallback_kind="llama-fit-params",
    )


def _fit_process_failure_kind(process: object) -> PlacementFailureKind | None:
    # Typed this way to keep the lifecycle mapping independent of persistence DTOs.
    from llama_profile_lab.execution.process import ProcessResult

    if not isinstance(process, ProcessResult):
        raise TypeError("expected ProcessResult")
    if process.interrupted:
        return "interrupted"
    if process.cancelled:
        return "cancelled"
    if process.timed_out:
        return "timeout"
    if process.exit_code == 0:
        return None
    return "fit_failed"


def _append_error(stderr: str, error: str) -> str:
    return error if not stderr else f"{stderr.rstrip()}\n{error}"
