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
from llama_profile_lab.db import (
    CandidateRepository,
    Database,
    ExperimentRepository,
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
) -> tuple[tuple[LauncherArgChange, ...], dict[str, JsonScalar]]:
    source = profile.candidate
    if candidate.model != source.model:
        raise PromotionError("promotion cannot change target or draft model identity")
    if candidate.placement.mode != source.placement.mode:
        raise PromotionError("promotion cannot change launcher placement mode")
    if candidate.placement.constraints.devices != source.placement.constraints.devices:
        raise PromotionError("launcher promotion does not yet encode device-list changes")
    if candidate.placement.constraints.tensor_split != source.placement.constraints.tensor_split:
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
