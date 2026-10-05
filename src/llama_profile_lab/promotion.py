"""Validated Candidate promotion proposals for llama-profile-launcher."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from difflib import unified_diff
from typing import Any

from llama_profile_lab.api.profiles import (
    LauncherProfile,
    LauncherProfileProvider,
)
from llama_profile_lab.analysis import DeploymentAnalysisService
from llama_profile_lab.db import (
    CandidateRepository,
    Database,
    DeploymentCandidateRepository,
    DeploymentPlacementRepository,
    DeploymentPromotionRepository,
    EnvironmentRepository,
    ExperimentRepository,
    PlacementRepository,
    ServerValidationRepository,
)
from llama_profile_lab.domain import Candidate
from llama_profile_lab.domain.base import JsonScalar


class PromotionError(RuntimeError):
    """Raised when a Candidate cannot produce a safe launcher proposal."""


@dataclass(frozen=True, slots=True)
class LauncherArgChange:
    """One semantic Candidate change represented by one launcher argument."""

    path: str
    argument: str
    before: JsonScalar
    after: JsonScalar


@dataclass(frozen=True, slots=True)
class PromotionProposal:
    """One persisted, non-destructive launcher promotion proposal."""

    id: str
    experiment_id: str
    candidate_id: str
    source_profile: str
    changes: tuple[LauncherArgChange, ...]
    patch: str
    source_snapshot: dict[str, Any]
    proposed_snapshot: dict[str, Any]
    validation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DeploymentPromotionSource:
    """Source experiment/profile provenance for one deployment instance."""

    instance_id: str
    experiment_id: str
    source_profile_id: str | None = None


@dataclass(frozen=True, slots=True)
class DeploymentPromotionProposal:
    """One coordinated, review-only launcher proposal for a joint placement."""

    id: str
    base_deployment_candidate_id: str
    deployment_candidate_id: str
    deployment_placement_id: str
    sources: tuple[dict[str, Any], ...]
    changes: tuple[dict[str, Any], ...]
    patch: str
    source_snapshot: dict[str, Any]
    proposed_snapshot: dict[str, Any]
    evidence: dict[str, Any]


class PromotionService:
    """Generate and persist launcher patches for server-validated Candidates."""

    def __init__(
        self,
        database: Database,
        profiles: LauncherProfileProvider,
    ) -> None:
        self.database = database
        self.profiles = profiles

    def propose(
        self,
        experiment_id: str,
        candidate_id: str,
        *,
        source_profile_id: str | None = None,
    ) -> PromotionProposal:
        with self.database.session() as connection:
            experiments = ExperimentRepository(connection)
            candidates = CandidateRepository(connection)
            experiment = experiments.get(experiment_id)
            if experiment is None:
                raise PromotionError(f"experiment not found: {experiment_id}")
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise PromotionError(f"Candidate not found: {candidate_id}")
            linked = connection.execute(
                """
                SELECT 1
                FROM experiment_candidate
                WHERE experiment_id = ? AND candidate_id = ?
                """,
                (experiment_id, candidate_id),
            ).fetchone()
            if linked is None:
                raise PromotionError(
                    f"Candidate {candidate_id} is not part of experiment {experiment_id}"
                )

            evaluations = ServerValidationRepository(connection).evaluations(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
            )
            completed_validations = tuple(
                item
                for item in evaluations
                if item.stage == "server-validated" and item.decision == "completed"
            )
            if not completed_validations:
                raise PromotionError(
                    "Candidate must have a completed server validation before promotion"
                )
            benchmarks = ServerValidationRepository(connection).benchmarks_for_candidate(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                completed_only=True,
            )
            base_candidate = candidates.get(experiment.base_candidate_id)
            if base_candidate is None:
                raise PromotionError("experiment base Candidate is missing")

        profile_id = source_profile_id or _launcher_profile_id(base_candidate)
        source_snapshot = self.profiles.snapshot()
        source_profile = self.profiles.resolve_snapshot(source_snapshot, profile_id)
        if source_profile is None:
            raise PromotionError(f"launcher profile not found: {profile_id}")
        if source_profile.candidate != base_candidate:
            raise PromotionError(
                "launcher source profile changed since experiment creation; "
                "create a new experiment from the current profile before promotion"
            )

        changes, updates = _launcher_changes(source_profile, candidate)
        proposed_snapshot = _apply_model_arg_updates(
            source_snapshot,
            profile_id=profile_id,
            updates=updates,
        )
        patch = _render_patch(source_snapshot, proposed_snapshot)
        validation = {
            "evaluations": [
                {
                    "id": item.id,
                    "decision": item.decision,
                    "reason": item.reason,
                    "metrics": dict(item.metrics),
                    "created_at": item.created_at,
                }
                for item in completed_validations
            ],
            "benchmarks": [
                {
                    "id": item.id,
                    "server_run_id": item.server_run_id,
                    "workload_case_id": item.workload_case_id,
                    "category": item.category,
                    "avg_prompt_ts": item.avg_prompt_ts,
                    "avg_pred_ts": item.avg_pred_ts,
                    "avg_latency_ms": item.avg_latency_ms,
                    "draft_n": item.draft_n,
                    "accepted_n": item.accepted_n,
                    "accept_rate": item.accept_rate,
                    "created_at": item.created_at,
                }
                for item in benchmarks
            ],
        }

        with self.database.session() as connection:
            identifier = ServerValidationRepository(connection).add_evaluation(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                stage="promotion",
                decision="proposed",
                metrics={
                    "source_profile": profile_id,
                    "changes": [asdict(item) for item in changes],
                    "source_profile_snapshot": source_snapshot,
                    "proposed_profile_snapshot": proposed_snapshot,
                    "validation": validation,
                    "patch": patch,
                },
            )

        return PromotionProposal(
            id=identifier,
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            source_profile=profile_id,
            changes=changes,
            patch=patch,
            source_snapshot=source_snapshot,
            proposed_snapshot=proposed_snapshot,
            validation=validation,
        )


class DeploymentPromotionService:
    """Generate one coordinated proposal for every instance in a joint placement."""

    def __init__(
        self,
        database: Database,
        profiles: LauncherProfileProvider,
    ) -> None:
        self.database = database
        self.profiles = profiles

    def propose(
        self,
        base_deployment_candidate_id: str,
        deployment_placement_id: str,
        *,
        sources: tuple[DeploymentPromotionSource, ...],
    ) -> DeploymentPromotionProposal:
        source_by_instance = {item.instance_id: item for item in sources}
        if len(source_by_instance) != len(sources):
            raise PromotionError("deployment promotion sources must have unique instance IDs")

        with self.database.session() as connection:
            deployments = DeploymentCandidateRepository(connection)
            placement_repository = DeploymentPlacementRepository(connection)
            candidates = CandidateRepository(connection)
            experiments = ExperimentRepository(connection)
            environment = EnvironmentRepository(connection)
            placements = PlacementRepository(connection)

            placement = placement_repository.get(deployment_placement_id)
            placement_record = placement_repository.record(deployment_placement_id)
            if placement is None or placement_record is None:
                raise PromotionError(
                    f"deployment placement not found: {deployment_placement_id}"
                )
            if not _deployment_contains_placement(
                connection,
                base_deployment_candidate_id,
                deployment_placement_id,
            ):
                raise PromotionError(
                    "deployment placement does not belong to this deployment workflow"
                )
            deployment = deployments.get(placement.deployment_candidate_id)
            if deployment is None:
                raise PromotionError("deployment Candidate is missing")

            expected_ids = {item.instance_id for item in deployment.instances}
            if set(source_by_instance) != expected_ids:
                missing = sorted(expected_ids - set(source_by_instance))
                extra = sorted(set(source_by_instance) - expected_ids)
                raise PromotionError(
                    "deployment promotion requires source provenance for every "
                    f"instance; missing={missing}, extra={extra}"
                )

            run_row = connection.execute(
                """
                SELECT id
                FROM deployment_run
                WHERE deployment_placement_id = ?
                  AND status = 'completed'
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (deployment_placement_id,),
            ).fetchone()
            if run_row is None:
                raise PromotionError(
                    "deployment placement must have a completed joint run before promotion"
                )
            deployment_run_id = str(run_row["id"])
            phase_rows = connection.execute(
                """
                SELECT id, phase, status, quality, correctness_valid,
                       combined_prompt_tps, combined_decode_tps,
                       min_retention, failure_kind, failure_details_json
                FROM deployment_workload_run
                WHERE deployment_run_id = ?
                ORDER BY created_at, id
                """,
                (deployment_run_id,),
            ).fetchall()
            completed_phases = {
                str(row["phase"])
                for row in phase_rows
                if str(row["status"]) == "completed"
                and bool(row["correctness_valid"])
            }
            expected_phases = set(deployment.workload_mix.phases)
            if not expected_phases.issubset(completed_phases):
                missing_phases = sorted(expected_phases - completed_phases)
                raise PromotionError(
                    "deployment promotion requires correctness-valid completed "
                    f"workloads for every configured phase; missing={missing_phases}"
                )

            instance_placements = {
                item.instance_id: item.resolved_placement_id
                for item in placement.instance_placements
            }
            if set(instance_placements) != expected_ids:
                raise PromotionError(
                    "deployment placement does not resolve every deployment instance"
                )

            source_snapshot = self.profiles.snapshot()
            proposed_snapshot = deepcopy(source_snapshot)
            source_payloads: list[dict[str, Any]] = []
            change_payloads: list[dict[str, Any]] = []
            binary_evidence: list[dict[str, Any]] = []
            placement_evidence: list[dict[str, Any]] = []

            for instance in deployment.instances:
                source = source_by_instance[instance.instance_id]
                experiment = experiments.get(source.experiment_id)
                if experiment is None:
                    raise PromotionError(
                        f"source experiment not found for {instance.instance_id}: "
                        f"{source.experiment_id}"
                    )
                linked = connection.execute(
                    """
                    SELECT 1
                    FROM experiment_candidate
                    WHERE experiment_id = ? AND candidate_id = ?
                    """,
                    (source.experiment_id, instance.candidate_id),
                ).fetchone()
                if linked is None:
                    raise PromotionError(
                        f"Candidate {instance.candidate_id} is not part of source "
                        f"experiment {source.experiment_id}"
                    )
                base_candidate = candidates.get(experiment.base_candidate_id)
                selected_candidate = candidates.get(instance.candidate_id)
                if base_candidate is None or selected_candidate is None:
                    raise PromotionError(
                        f"source Candidate data is missing for {instance.instance_id}"
                    )

                profile_id = source.source_profile_id or _launcher_profile_id(
                    base_candidate
                )
                source_profile = self.profiles.resolve_snapshot(
                    source_snapshot,
                    profile_id,
                )
                if source_profile is None:
                    raise PromotionError(
                        f"launcher profile not found for {instance.instance_id}: "
                        f"{profile_id}"
                    )
                if source_profile.candidate != base_candidate:
                    raise PromotionError(
                        f"launcher source profile {profile_id} changed since source "
                        f"experiment {source.experiment_id}; create a new experiment "
                        "before coordinated promotion"
                    )

                resolved_id = instance_placements[instance.instance_id]
                resolved = placements.get(resolved_id)
                if resolved is None:
                    raise PromotionError(
                        f"resolved placement disappeared: {resolved_id}"
                    )
                changes, updates = _deployment_launcher_changes(
                    source_profile,
                    selected_candidate,
                    resolved,
                )
                proposed_snapshot = _apply_model_arg_updates(
                    proposed_snapshot,
                    profile_id=profile_id,
                    updates=updates,
                )

                source_payloads.append(
                    {
                        "instance_id": instance.instance_id,
                        "experiment_id": source.experiment_id,
                        "candidate_id": instance.candidate_id,
                        "source_profile_id": profile_id,
                    }
                )
                change_payloads.append(
                    {
                        "instance_id": instance.instance_id,
                        "candidate_id": instance.candidate_id,
                        "source_profile_id": profile_id,
                        "changes": [asdict(item) for item in changes],
                    }
                )
                binary = environment.get_binary(instance.binary_id)
                if binary is None:
                    raise PromotionError(
                        f"registered server binary disappeared: {instance.binary_id}"
                    )
                helper = connection.execute(
                    """
                    SELECT me.helper_binary_id, b.sha256
                    FROM memory_estimate AS me
                    JOIN binary AS b ON b.id = me.helper_binary_id
                    WHERE me.candidate_id = ? AND me.host_id = ?
                    ORDER BY me.created_at DESC, me.id DESC
                    LIMIT 1
                    """,
                    (instance.candidate_id, placement.host_id),
                ).fetchone()
                binary_evidence.append(
                    {
                        "instance_id": instance.instance_id,
                        "server_binary_id": binary.id,
                        "server_sha256": binary.sha256,
                        "helper_binary_id": (
                            None if helper is None else str(helper["helper_binary_id"])
                        ),
                        "helper_sha256": (
                            None if helper is None else str(helper["sha256"])
                        ),
                        "model_artifact_id": instance.model_artifact_id,
                    }
                )
                placement_evidence.append(
                    {
                        "instance_id": instance.instance_id,
                        "resolved_placement_id": resolved.id,
                        "production_context_size": resolved.production_context_size,
                        "n_gpu_layers": resolved.n_gpu_layers,
                        "n_cpu_moe": resolved.n_cpu_moe,
                        "split_mode": resolved.split_mode,
                        "main_gpu": resolved.main_gpu,
                        "devices": resolved.devices,
                        "tensor_split": resolved.tensor_split,
                        "override_tensor": resolved.override_tensor,
                    }
                )

            phase_evidence = [
                {
                    "id": str(row["id"]),
                    "phase": str(row["phase"]),
                    "status": str(row["status"]),
                    "quality": row["quality"],
                    "correctness_valid": bool(row["correctness_valid"]),
                    "combined_prompt_tps": row["combined_prompt_tps"],
                    "combined_decode_tps": row["combined_decode_tps"],
                    "min_retention": row["min_retention"],
                    "failure_kind": row["failure_kind"],
                }
                for row in phase_rows
            ]
            projected_memory = [
                {
                    "instance_id": item.instance_id,
                    "device_id": item.device_id,
                    "model_bytes": item.model_bytes,
                    "context_bytes": item.context_bytes,
                    "compute_bytes": item.compute_bytes,
                    "total_bytes": item.total_bytes,
                    "device_total_bytes": item.device_total_bytes,
                    "device_free_bytes": item.device_free_bytes,
                    "source": item.source,
                }
                for item in placement_repository.memory(deployment_placement_id)
            ]
            allocations = [
                {
                    "device_id": item.device_id,
                    "projected_bytes": item.projected_bytes,
                    "reserved_margin_bytes": item.reserved_margin_bytes,
                    "device_total_bytes": item.device_total_bytes,
                    "projected_free_bytes": item.projected_free_bytes,
                }
                for item in placement_repository.allocations(deployment_placement_id)
            ]

        runtime_memory = DeploymentAnalysisService(self.database).memory_matrix(
            deployment_placement_id,
            deployment_run_id=deployment_run_id,
        ).model_dump(mode="json")
        evidence = {
            "deployment_run_id": deployment_run_id,
            "binaries": binary_evidence,
            "placements": placement_evidence,
            "projected_memory": projected_memory,
            "device_allocations": allocations,
            "runtime_memory": runtime_memory,
            "concurrent_validation": phase_evidence,
        }
        patch = _render_patch(source_snapshot, proposed_snapshot)

        with self.database.session() as connection:
            identifier = DeploymentPromotionRepository(connection).create(
                base_deployment_candidate_id=base_deployment_candidate_id,
                deployment_candidate_id=placement.deployment_candidate_id,
                deployment_placement_id=deployment_placement_id,
                sources=source_payloads,
                changes=change_payloads,
                source_snapshot=source_snapshot,
                proposed_snapshot=proposed_snapshot,
                evidence=evidence,
                patch=patch,
            )

        return DeploymentPromotionProposal(
            id=identifier,
            base_deployment_candidate_id=base_deployment_candidate_id,
            deployment_candidate_id=placement.deployment_candidate_id,
            deployment_placement_id=deployment_placement_id,
            sources=tuple(source_payloads),
            changes=tuple(change_payloads),
            patch=patch,
            source_snapshot=source_snapshot,
            proposed_snapshot=proposed_snapshot,
            evidence=evidence,
        )


def _deployment_contains_placement(
    connection: Any,
    base_deployment_candidate_id: str,
    deployment_placement_id: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM deployment_placement AS dp
        WHERE dp.id = ?
          AND (
              dp.deployment_candidate_id = ?
              OR EXISTS (
                  SELECT 1
                  FROM deployment_plan AS plan
                  JOIN deployment_plan_case AS pc
                    ON pc.deployment_plan_id = plan.id
                  WHERE plan.base_deployment_candidate_id = ?
                    AND pc.deployment_placement_id = dp.id
              )
          )
        LIMIT 1
        """,
        (
            deployment_placement_id,
            base_deployment_candidate_id,
            base_deployment_candidate_id,
        ),
    ).fetchone()
    return row is not None


def _launcher_profile_id(candidate: Candidate) -> str:
    prefix = "launcher-profile:"
    target = candidate.model.target_model_id
    if not target.startswith(prefix) or len(target) == len(prefix):
        raise PromotionError(
            "experiment base Candidate does not identify a launcher profile"
        )
    return target[len(prefix) :]


def _launcher_changes(
    profile: LauncherProfile,
    candidate: Candidate,
    *,
    allow_placement_lists: bool = False,
) -> tuple[tuple[LauncherArgChange, ...], dict[str, JsonScalar]]:
    source = profile.candidate
    if candidate.model != source.model:
        raise PromotionError("promotion cannot change target or draft model identity")
    if candidate.placement.mode != source.placement.mode:
        raise PromotionError("promotion cannot change launcher placement mode")
    if not allow_placement_lists:
        if candidate.placement.constraints.devices != source.placement.constraints.devices:
            raise PromotionError("launcher promotion does not yet encode device-list changes")
        if (
            candidate.placement.constraints.tensor_split
            != source.placement.constraints.tensor_split
        ):
            raise PromotionError("launcher promotion does not yet encode tensor-split changes")
        if (
            candidate.placement.constraints.override_tensor
            != source.placement.constraints.override_tensor
        ):
            raise PromotionError(
                "launcher promotion does not yet encode override-tensor changes"
            )
    if source.placement.fit is not None and candidate.placement.fit is not None:
        if source.placement.fit.min_context != candidate.placement.fit.min_context:
            raise PromotionError(
                "launcher promotion cannot represent placement.fit.min_context"
            )

    changes: list[LauncherArgChange] = []
    updates: dict[str, JsonScalar] = {}

    def add(
        path: str,
        argument: str,
        before: JsonScalar,
        after: JsonScalar,
        encoded_after: JsonScalar | None = None,
    ) -> None:
        if before == after:
            return
        changes.append(
            LauncherArgChange(
                path=path,
                argument=argument,
                before=before,
                after=after,
            )
        )
        updates[argument] = after if encoded_after is None else encoded_after

    add("context.size", "--ctx-size", source.context.size, candidate.context.size)
    add(
        "context.cache_type_k",
        "--cache-type-k",
        source.context.cache_type_k,
        candidate.context.cache_type_k,
    )
    add(
        "context.cache_type_v",
        "--cache-type-v",
        source.context.cache_type_v,
        candidate.context.cache_type_v,
    )
    add(
        "context.kv_offload",
        "--no-kv-offload",
        source.context.kv_offload,
        candidate.context.kv_offload,
        not candidate.context.kv_offload,
    )
    add(
        "context.kv_unified",
        "--no-kv-unified",
        source.context.kv_unified,
        candidate.context.kv_unified,
        not candidate.context.kv_unified,
    )
    add(
        "compute.flash_attn",
        "--flash-attn",
        source.compute.flash_attn,
        candidate.compute.flash_attn,
    )
    add(
        "compute.batch_size",
        "--batch-size",
        source.compute.batch_size,
        candidate.compute.batch_size,
    )
    add(
        "compute.ubatch_size",
        "--ubatch-size",
        source.compute.ubatch_size,
        candidate.compute.ubatch_size,
    )
    add(
        "compute.threads",
        "--threads",
        source.compute.threads,
        candidate.compute.threads,
    )
    add(
        "compute.load_mode",
        "--load-mode",
        source.compute.load_mode,
        candidate.compute.load_mode,
    )
    add(
        "compute.lazy_mode",
        "--lazy-mode",
        source.compute.lazy_mode,
        candidate.compute.lazy_mode,
    )
    add(
        "compute.repack",
        "--no-repack",
        source.compute.repack,
        candidate.compute.repack,
        not candidate.compute.repack,
    )
    add(
        "compute.no_host",
        "--no-host",
        source.compute.no_host,
        candidate.compute.no_host,
    )
    add(
        "compute.no_op_offload",
        "--no-op-offload",
        source.compute.no_op_offload,
        candidate.compute.no_op_offload,
    )

    if source.placement.fit is not None and candidate.placement.fit is not None:
        add(
            "placement.fit.target_mib",
            "--fit-target",
            source.placement.fit.target_mib,
            candidate.placement.fit.target_mib,
        )
    add(
        "placement.constraints.n_gpu_layers",
        "--n-gpu-layers",
        source.placement.constraints.n_gpu_layers,
        candidate.placement.constraints.n_gpu_layers,
    )
    add(
        "placement.constraints.n_cpu_moe",
        "--n-cpu-moe",
        source.placement.constraints.n_cpu_moe,
        candidate.placement.constraints.n_cpu_moe,
    )
    add(
        "placement.constraints.split_mode",
        "--split-mode",
        source.placement.constraints.split_mode,
        candidate.placement.constraints.split_mode,
    )
    add(
        "placement.constraints.main_gpu",
        "--main-gpu",
        source.placement.constraints.main_gpu,
        candidate.placement.constraints.main_gpu,
    )
    add(
        "server.parallel",
        "--parallel",
        source.server.parallel,
        candidate.server.parallel,
    )
    add(
        "speculative.type",
        "--spec-type",
        source.speculative.type,
        candidate.speculative.type,
    )
    add(
        "speculative.draft_n_max",
        "--spec-draft-n-max",
        source.speculative.draft_n_max,
        candidate.speculative.draft_n_max,
    )

    source_extra = {item.name: item.value for item in source.extra_args}
    target_extra = {item.name: item.value for item in candidate.extra_args}
    removed_extra = set(source_extra) - set(target_extra)
    if removed_extra:
        raise PromotionError(
            "launcher promotion cannot safely remove extra arguments: "
            + ", ".join(sorted(removed_extra))
        )
    for name in sorted(target_extra):
        before = source_extra.get(name)
        after = target_extra[name]
        if before == after:
            continue
        if isinstance(after, tuple):
            raise PromotionError(
                f"launcher promotion cannot encode tuple-valued extra argument {name}"
            )
        changes.append(
            LauncherArgChange(
                path=f"extra_args.{name}",
                argument=name,
                before=before if not isinstance(before, tuple) else None,
                after=after,
            )
        )
        updates[name] = after

    return tuple(changes), updates


def _deployment_launcher_changes(
    profile: LauncherProfile,
    candidate: Candidate,
    resolved: Any,
) -> tuple[tuple[LauncherArgChange, ...], dict[str, JsonScalar]]:
    promotion_candidate = candidate.model_copy(
        update={"placement": profile.candidate.placement}
    )
    changes, updates = _launcher_changes(
        profile,
        promotion_candidate,
        allow_placement_lists=True,
    )
    change_list = list(changes)
    source = profile.candidate.placement.constraints

    def set_arg(
        path: str,
        argument: str,
        before: JsonScalar,
        after: JsonScalar,
    ) -> None:
        if before == after:
            return
        change_list.append(
            LauncherArgChange(
                path=path,
                argument=argument,
                before=before,
                after=after,
            )
        )
        updates[argument] = after

    set_arg(
        "placement.resolved.production_context_size",
        "--ctx-size",
        profile.candidate.context.size,
        resolved.production_context_size,
    )
    set_arg(
        "placement.resolved.n_gpu_layers",
        "--n-gpu-layers",
        source.n_gpu_layers,
        resolved.n_gpu_layers,
    )
    set_arg(
        "placement.resolved.n_cpu_moe",
        "--n-cpu-moe",
        source.n_cpu_moe,
        resolved.n_cpu_moe,
    )
    set_arg(
        "placement.resolved.split_mode",
        "--split-mode",
        source.split_mode,
        resolved.split_mode,
    )
    set_arg(
        "placement.resolved.main_gpu",
        "--main-gpu",
        source.main_gpu,
        resolved.main_gpu,
    )
    devices = (
        "auto"
        if resolved.devices == "auto"
        else ",".join(str(item) for item in resolved.devices)
    )
    source_devices = (
        source.devices
        if isinstance(source.devices, str)
        else ",".join(str(item) for item in source.devices)
    )
    set_arg(
        "placement.resolved.devices",
        "--device",
        source_devices,
        devices,
    )
    tensor_split = (
        None
        if resolved.tensor_split is None
        else ",".join(str(item) for item in resolved.tensor_split)
    )
    source_tensor_split = (
        None
        if source.tensor_split is None
        else ",".join(str(item) for item in source.tensor_split)
    )
    set_arg(
        "placement.resolved.tensor_split",
        "--tensor-split",
        source_tensor_split,
        tensor_split,
    )
    override_tensor = (
        None
        if not resolved.override_tensor
        else ",".join(resolved.override_tensor)
    )
    source_override = (
        None
        if not source.override_tensor
        else ",".join(source.override_tensor)
    )
    set_arg(
        "placement.resolved.override_tensor",
        "--override-tensor",
        source_override,
        override_tensor,
    )
    return tuple(change_list), updates


def _apply_model_arg_updates(
    source_snapshot: dict[str, Any],
    *,
    profile_id: str,
    updates: dict[str, JsonScalar],
) -> dict[str, Any]:
    proposed = deepcopy(source_snapshot)
    models = proposed.get("models")
    if not isinstance(models, dict):
        raise PromotionError("launcher models must be an object")
    model = models.get(profile_id)
    if not isinstance(model, dict):
        raise PromotionError(f"launcher profile not found: {profile_id}")
    raw_args = model.get("args", {})
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        raise PromotionError(f"launcher model {profile_id!r} args must be an object")
    model_args = dict(raw_args)
    model_args.update(updates)
    model["args"] = model_args
    return proposed


def _render_patch(
    source_snapshot: dict[str, Any],
    proposed_snapshot: dict[str, Any],
) -> str:
    before = json.dumps(
        source_snapshot,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).splitlines(keepends=True)
    after = json.dumps(
        proposed_snapshot,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).splitlines(keepends=True)
    return "".join(
        unified_diff(
            before,
            after,
            fromfile="a/launcher-config.json",
            tofile="b/launcher-config.json",
        )
    )
