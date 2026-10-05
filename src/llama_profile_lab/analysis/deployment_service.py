"""Read-only analysis over persisted multi-model deployment results."""

from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from llama_profile_lab.analysis.models import (
    DeploymentAnalysisFilter,
    DeploymentComparison,
    DeploymentInterferenceMember,
    DeploymentInterferenceView,
    DeploymentMatrixCell,
    DeploymentMatrixFacet,
    DeploymentMatrixProjection,
    DeploymentMemoryMatrix,
    DeploymentMemoryRow,
    DeploymentMetricConstraint,
    DeploymentMetricDelta,
    DeploymentParetoObjective,
    DeploymentParetoPoint,
    DeploymentParetoResult,
)
from llama_profile_lab.db import (
    CandidateRepository,
    ConcurrentWorkloadRepository,
    Database,
    DeploymentCandidateRepository,
    DeploymentPlacementRepository,
    DeploymentRunRepository,
    PlacementRepository,
)
from llama_profile_lab.db.records import (
    DeploymentDeviceAllocationRecord,
    DeploymentWorkloadMemberRecord,
    PlacementDeviceMemoryRecord,
    ResolvedPlacementRecord,
)
from llama_profile_lab.domain import (
    Candidate,
    ConcurrentWorkloadCase,
    DeploymentCandidate,
    GpuTelemetrySample,
)
from llama_profile_lab.domain.base import JsonScalar


class DeploymentAnalysisError(RuntimeError):
    """Raised when deployment analysis is missing evidence or is ambiguous."""


@dataclass(frozen=True, slots=True)
class _PlacementContext:
    placement_id: str
    deployment_candidate_id: str
    deployment: DeploymentCandidate
    candidates: Mapping[str, Candidate]
    resolved: Mapping[str, ResolvedPlacementRecord]
    backends: Mapping[str, tuple[str, ...]]
    memory: tuple[PlacementDeviceMemoryRecord, ...]
    allocations: tuple[DeploymentDeviceAllocationRecord, ...]


@dataclass(frozen=True, slots=True)
class _Observation:
    deployment_run_id: str
    deployment_run_created_at: str
    deployment_status: str
    deployment_quality: str | None
    deployment_failure_kind: str | None
    workload_run_id: str | None
    workload_case_id: str | None
    phase: str | None
    workload_status: str | None
    workload_quality: str | None
    correctness_valid: bool | None
    combined_prompt_tps: float | None
    combined_decode_tps: float | None
    min_retention: float | None
    context: _PlacementContext
    workload: ConcurrentWorkloadCase | None
    members: tuple[DeploymentWorkloadMemberRecord, ...]
    gpu_samples: tuple[tuple[int, tuple[GpuTelemetrySample, ...]], ...]


@dataclass(frozen=True, slots=True)
class _RuntimeSummary:
    peak_used_by_device: Mapping[str, int]
    min_free_by_device: Mapping[str, int]
    total_power_avg_w: float | None
    total_power_peak_w: float | None
    total_power_sum_w: float
    total_power_sample_count: int


class DeploymentAnalysisService:
    """Deployment projections, interference, comparison, Pareto, and export."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def metric_names(self) -> tuple[str, ...]:
        """Return the stable deployment metric namespace."""
        return (
            "deployment.combined_tg_tps",
            "deployment.combined_pp_tps",
            "deployment.min_retention",
            "deployment.total_validated_context_tokens",
            "deployment.min_device_headroom_bytes",
            "deployment.min_device_projected_headroom_bytes",
            "deployment.min_device_runtime_headroom_bytes",
            "deployment.total_power_avg_w",
            "deployment.total_power_peak_w",
            "deployment.device.<id>.headroom_bytes",
            "deployment.device.<id>.projected_headroom_bytes",
            "deployment.device.<id>.runtime_headroom_bytes",
            "deployment.instance.<id>.pp_tps",
            "deployment.instance.<id>.tg_tps",
            "deployment.instance.<id>.retention",
            "deployment.instance.<id>.latency_ms",
            "deployment.instance.<id>.latency_increase_pct",
        )

    def matrix(
        self,
        *,
        x_path: str,
        y_path: str,
        metric: str,
        filters: Sequence[DeploymentAnalysisFilter] = (),
        facet_path: str | None = None,
    ) -> DeploymentMatrixProjection:
        """Project one exact deployment/workload slice into sparse X/Y cells."""
        observations = [
            item
            for item in self._load()
            if _valid(item) and _matches_filters(item, filters)
        ]
        if not observations:
            raise DeploymentAnalysisError(
                "no valid deployment observations match the matrix filters"
            )

        groups: dict[
            tuple[JsonScalar, JsonScalar, JsonScalar],
            list[_Observation],
        ] = defaultdict(list)
        for item in observations:
            facet = _resolve_path(item, facet_path) if facet_path else None
            groups[
                (
                    facet,
                    _resolve_path(item, x_path),
                    _resolve_path(item, y_path),
                )
            ].append(item)

        facets: dict[JsonScalar, list[DeploymentMatrixCell]] = defaultdict(list)
        for (facet, x_value, y_value), group in groups.items():
            placements = {item.context.placement_id for item in group}
            workloads = {_workload_signature(item) for item in group}
            if len(placements) != 1 or len(workloads) != 1:
                raise DeploymentAnalysisError(
                    "deployment matrix cell collapses hidden placement/workload "
                    "coordinates; add exact filters or a facet"
                )
            value = _evaluate_metric(metric, group)
            if value is None:
                continue
            first = group[0]
            facets[facet].append(
                DeploymentMatrixCell(
                    x=x_value,
                    y=y_value,
                    value=value,
                    deployment_candidate_id=first.context.deployment_candidate_id,
                    deployment_placement_id=first.context.placement_id,
                    workload_case_id=first.workload_case_id,
                    observation_count=len(group),
                )
            )

        if not facets:
            raise DeploymentAnalysisError(
                f"metric {metric!r} has no values for the selected observations"
            )
        x_values = _sorted_scalars(
            {cell.x for values in facets.values() for cell in values}
        )
        y_values = _sorted_scalars(
            {cell.y for values in facets.values() for cell in values}
        )
        return DeploymentMatrixProjection(
            x_path=x_path,
            y_path=y_path,
            metric=metric,
            facet_path=facet_path,
            x_values=x_values,
            y_values=y_values,
            facets=tuple(
                DeploymentMatrixFacet(
                    value=facet,
                    cells=tuple(
                        sorted(
                            values,
                            key=lambda item: (
                                _scalar_sort_key(item.y),
                                _scalar_sort_key(item.x),
                            ),
                        )
                    ),
                )
                for facet, values in sorted(
                    facets.items(),
                    key=lambda item: _scalar_sort_key(item[0]),
                )
            ),
        )

    def memory_matrix(
        self,
        deployment_placement_id: str,
        *,
        deployment_run_id: str | None = None,
    ) -> DeploymentMemoryMatrix:
        """Return projected memory plus separately labeled runtime peak evidence."""
        observations = [
            item
            for item in self._load()
            if item.context.placement_id == deployment_placement_id
        ]
        if not observations:
            raise DeploymentAnalysisError(
                f"deployment placement has no analysis evidence: "
                f"{deployment_placement_id}"
            )
        context = observations[0].context
        run_id = deployment_run_id
        if run_id is None:
            candidates = sorted(
                (
                    item
                    for item in observations
                    if item.gpu_samples
                ),
                key=lambda item: (
                    item.deployment_run_created_at,
                    item.deployment_run_id,
                ),
                reverse=True,
            )
            run_id = (
                candidates[0].deployment_run_id
                if candidates
                else None
            )
        run_observation = None
        if run_id is not None:
            run_observation = next(
                (
                    item
                    for item in observations
                    if item.deployment_run_id == run_id
                ),
                None,
            )
            if run_observation is None:
                raise DeploymentAnalysisError(
                    "deployment run does not belong to requested placement"
                )

        devices = tuple(item.device_id for item in context.allocations)
        rows: list[DeploymentMemoryRow] = []
        for instance_id in sorted(context.candidates):
            entries = {
                item.device_id: item
                for item in context.memory
                if item.instance_id == instance_id
            }
            for category in ("model", "context", "compute"):
                rows.append(
                    DeploymentMemoryRow(
                        key=f"{instance_id}.{category}",
                        source="projected",
                        values={
                            device: (
                                None
                                if device not in entries
                                else int(
                                    getattr(
                                        entries[device],
                                        f"{category}_bytes",
                                    )
                                )
                            )
                            for device in devices
                        },
                    )
                )

        allocations = {
            item.device_id: item for item in context.allocations
        }
        rows.extend(
            (
                DeploymentMemoryRow(
                    key="reserved",
                    source="projected",
                    values={
                        device: allocations[device].reserved_margin_bytes
                        for device in devices
                    },
                ),
                DeploymentMemoryRow(
                    key="projected_free",
                    source="projected",
                    values={
                        device: allocations[device].projected_free_bytes
                        for device in devices
                    },
                ),
            )
        )
        runtime = (
            _runtime_summary(run_observation)
            if run_observation is not None
            else _RuntimeSummary({}, {}, None, None, 0.0, 0)
        )
        rows.extend(
            (
                DeploymentMemoryRow(
                    key="runtime_peak",
                    source="runtime",
                    values={
                        device: runtime.peak_used_by_device.get(device)
                        for device in devices
                    },
                ),
                DeploymentMemoryRow(
                    key="runtime_free_min",
                    source="runtime",
                    values={
                        device: runtime.min_free_by_device.get(device)
                        for device in devices
                    },
                ),
            )
        )
        return DeploymentMemoryMatrix(
            deployment_placement_id=deployment_placement_id,
            deployment_run_id=run_id,
            devices=devices,
            rows=tuple(rows),
        )

    def interference(
        self,
        workload_run_id: str,
    ) -> DeploymentInterferenceView:
        """Return one phase with aggregate and per-instance interference."""
        observation = next(
            (
                item
                for item in self._load()
                if item.workload_run_id == workload_run_id
            ),
            None,
        )
        if observation is None or observation.phase is None:
            raise DeploymentAnalysisError(
                f"concurrent workload run not found: {workload_run_id}"
            )
        return DeploymentInterferenceView(
            deployment_run_id=observation.deployment_run_id,
            workload_run_id=workload_run_id,
            deployment_placement_id=observation.context.placement_id,
            phase=observation.phase,
            quality=observation.workload_quality,
            combined_prompt_tps=observation.combined_prompt_tps,
            combined_decode_tps=observation.combined_decode_tps,
            min_retention=observation.min_retention,
            members=tuple(
                DeploymentInterferenceMember(
                    instance_id=item.instance_id,
                    mode=item.mode,
                    native_tps=(
                        item.native_prompt_tps
                        if item.mode == "prefill"
                        else item.native_decode_tps
                    ),
                    overlap_tps=(
                        item.overlap_prompt_tps
                        if item.mode == "prefill"
                        else item.overlap_decode_tps
                    ),
                    standalone_tps=item.standalone_tps,
                    retention=item.retention,
                    throughput_loss_pct=item.throughput_loss_pct,
                    latency_ms=item.latency_ms,
                    baseline_latency_ms=item.baseline_latency_ms,
                    latency_increase_pct=item.latency_increase_pct,
                    correctness_valid=item.correctness_valid,
                )
                for item in observation.members
            ),
        )

    def compare(
        self,
        deployment_placement_id: str,
        baseline_placement_id: str,
        *,
        metrics: Sequence[str],
        filters: Sequence[DeploymentAnalysisFilter] = (),
    ) -> DeploymentComparison:
        """Compare placement metrics with candidate-minus-baseline signs."""
        if not metrics:
            raise DeploymentAnalysisError(
                "deployment comparison requires at least one metric"
            )
        observations = self._load()
        current = [
            item
            for item in observations
            if _valid(item)
            and item.context.placement_id == deployment_placement_id
            and _matches_filters(item, filters)
        ]
        baseline = [
            item
            for item in observations
            if _valid(item)
            and item.context.placement_id == baseline_placement_id
            and _matches_filters(item, filters)
        ]
        if not current:
            raise DeploymentAnalysisError(
                "candidate placement has no matching valid observations"
            )
        if not baseline:
            raise DeploymentAnalysisError(
                "baseline placement has no matching valid observations"
            )
        deltas: list[DeploymentMetricDelta] = []
        for metric in metrics:
            candidate_value = _evaluate_metric(metric, current)
            baseline_value = _evaluate_metric(metric, baseline)
            delta = None
            percent = None
            if candidate_value is not None and baseline_value is not None:
                delta = candidate_value - baseline_value
                if baseline_value != 0:
                    percent = delta / abs(baseline_value) * 100.0
            deltas.append(
                DeploymentMetricDelta(
                    metric=metric,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    delta=delta,
                    percent_delta=percent,
                )
            )
        return DeploymentComparison(
            deployment_placement_id=deployment_placement_id,
            baseline_placement_id=baseline_placement_id,
            deltas=tuple(deltas),
        )

    def pareto(
        self,
        *,
        objectives: Sequence[DeploymentParetoObjective],
        constraints: Sequence[DeploymentMetricConstraint] = (),
        filters: Sequence[DeploymentAnalysisFilter] = (),
        deployment_candidate_ids: Sequence[str] | None = None,
    ) -> DeploymentParetoResult:
        """Return non-dominated valid placements after metric constraints."""
        if not objectives:
            raise DeploymentAnalysisError(
                "at least one deployment Pareto objective is required"
            )
        keys = [item.key for item in objectives]
        if len(keys) != len(set(keys)):
            raise DeploymentAnalysisError(
                "deployment Pareto objective keys must be unique"
            )

        observations = self._load()
        candidate_scope = (
            None
            if deployment_candidate_ids is None
            else set(deployment_candidate_ids)
        )
        placement_ids = sorted(
            {
                item.context.placement_id
                for item in observations
                if (
                    candidate_scope is None
                    or item.context.deployment_candidate_id
                    in candidate_scope
                )
                and _matches_filters(
                    item,
                    filters,
                    missing_is_false=True,
                )
            }
        )
        vectors: list[DeploymentParetoPoint] = []
        excluded: dict[str, str] = {}

        for placement_id in placement_ids:
            subject = [
                item
                for item in observations
                if item.context.placement_id == placement_id
                and _valid(item)
                and _matches_filters(item, filters, missing_is_false=True)
            ]
            if not subject:
                excluded[placement_id] = (
                    "no valid completed concurrent observations"
                )
                continue

            constraint_failure = _constraint_failure(subject, constraints)
            if constraint_failure is not None:
                excluded[placement_id] = constraint_failure
                continue

            values: dict[str, float] = {}
            reason: str | None = None
            for objective in objectives:
                selected = [
                    item
                    for item in subject
                    if _matches_filters(
                        item,
                        objective.filters,
                        missing_is_false=True,
                    )
                ]
                if not selected:
                    reason = f"no data for objective {objective.key}"
                    break
                value = _evaluate_metric(objective.metric, selected)
                if value is None or not math.isfinite(value):
                    reason = (
                        f"metric {objective.metric} unavailable for "
                        f"{objective.key}"
                    )
                    break
                values[objective.key] = value
            if reason is not None:
                excluded[placement_id] = reason
                continue

            first = subject[0]
            vectors.append(
                DeploymentParetoPoint(
                    deployment_candidate_id=(
                        first.context.deployment_candidate_id
                    ),
                    deployment_placement_id=placement_id,
                    values=values,
                )
            )

        frontier = tuple(
            item
            for item in vectors
            if not any(
                _dominates(other, item, objectives)
                for other in vectors
                if other.deployment_placement_id
                != item.deployment_placement_id
            )
        )
        return DeploymentParetoResult(
            objectives=tuple(objectives),
            constraints=tuple(constraints),
            evaluated_count=len(vectors),
            frontier=frontier,
            excluded=excluded,
        )

    def export(
        self,
        *,
        format_name: str,
        filters: Sequence[DeploymentAnalysisFilter] = (),
        deployment_candidate_ids: Sequence[str] | None = None,
    ) -> str:
        """Export raw successful and failed deployment observations."""
        candidate_scope = (
            None
            if deployment_candidate_ids is None
            else set(deployment_candidate_ids)
        )
        rows = [
            _export_row(item)
            for item in self._load()
            if (
                candidate_scope is None
                or item.context.deployment_candidate_id
                in candidate_scope
            )
            and _matches_filters(
                item,
                filters,
                missing_is_false=True,
            )
        ]
        if format_name == "json":
            return json.dumps(rows, indent=2, sort_keys=True)
        if format_name != "csv":
            raise DeploymentAnalysisError(
                "deployment export format must be csv or json"
            )
        output = io.StringIO()
        fieldnames = _fieldnames(rows)
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return output.getvalue()

    def _load(self) -> tuple[_Observation, ...]:
        with self.database.session() as connection:
            rows = connection.execute(
                """
                SELECT dr.id AS deployment_run_id,
                       dr.created_at AS deployment_run_created_at,
                       dr.status AS deployment_status,
                       dr.quality AS deployment_quality,
                       dr.failure_kind AS deployment_failure_kind,
                       dr.deployment_placement_id,
                       dwr.id AS workload_run_id,
                       dwr.workload_case_id,
                       dwr.phase,
                       dwr.status AS workload_status,
                       dwr.quality AS workload_quality,
                       dwr.correctness_valid,
                       dwr.combined_prompt_tps,
                       dwr.combined_decode_tps,
                       dwr.min_retention,
                       dcw.definition_json
                FROM deployment_run AS dr
                LEFT JOIN deployment_workload_run AS dwr
                  ON dwr.deployment_run_id = dr.id
                LEFT JOIN deployment_concurrent_workload_case AS dcw
                  ON dcw.id = dwr.workload_case_id
                WHERE dr.deployment_placement_id IS NOT NULL
                ORDER BY dr.created_at, dr.id, dwr.created_at, dwr.id
                """
            ).fetchall()
            placements: dict[str, _PlacementContext] = {}
            run_samples: dict[
                str,
                tuple[tuple[int, tuple[GpuTelemetrySample, ...]], ...],
            ] = {}
            observations: list[_Observation] = []
            concurrent = ConcurrentWorkloadRepository(connection)
            deployment_runs = DeploymentRunRepository(connection)

            for row in rows:
                placement_id = str(row["deployment_placement_id"])
                context = placements.get(placement_id)
                if context is None:
                    context = _load_placement_context(
                        connection,
                        placement_id,
                    )
                    placements[placement_id] = context

                deployment_run_id = str(row["deployment_run_id"])
                gpu_samples = run_samples.get(deployment_run_id)
                if gpu_samples is None:
                    parsed: list[
                        tuple[int, tuple[GpuTelemetrySample, ...]]
                    ] = []
                    for sample in deployment_runs.gpu_samples(
                        deployment_run_id
                    ):
                        parsed.append(
                            (
                                sample.timestamp_ns,
                                tuple(
                                    GpuTelemetrySample.model_validate(
                                        dict(item)
                                    )
                                    for item in sample.gpus
                                ),
                            )
                        )
                    gpu_samples = tuple(parsed)
                    run_samples[deployment_run_id] = gpu_samples

                workload_run_id = row["workload_run_id"]
                workload = (
                    None
                    if row["definition_json"] is None
                    else ConcurrentWorkloadCase.model_validate_json(
                        str(row["definition_json"])
                    )
                )
                observations.append(
                    _Observation(
                        deployment_run_id=deployment_run_id,
                        deployment_run_created_at=str(
                            row["deployment_run_created_at"]
                        ),
                        deployment_status=str(row["deployment_status"]),
                        deployment_quality=row["deployment_quality"],
                        deployment_failure_kind=(
                            row["deployment_failure_kind"]
                        ),
                        workload_run_id=(
                            None
                            if workload_run_id is None
                            else str(workload_run_id)
                        ),
                        workload_case_id=(
                            None
                            if row["workload_case_id"] is None
                            else str(row["workload_case_id"])
                        ),
                        phase=row["phase"],
                        workload_status=row["workload_status"],
                        workload_quality=row["workload_quality"],
                        correctness_valid=(
                            None
                            if row["correctness_valid"] is None
                            else bool(row["correctness_valid"])
                        ),
                        combined_prompt_tps=(
                            None
                            if row["combined_prompt_tps"] is None
                            else float(row["combined_prompt_tps"])
                        ),
                        combined_decode_tps=(
                            None
                            if row["combined_decode_tps"] is None
                            else float(row["combined_decode_tps"])
                        ),
                        min_retention=(
                            None
                            if row["min_retention"] is None
                            else float(row["min_retention"])
                        ),
                        context=context,
                        workload=workload,
                        members=(
                            ()
                            if workload_run_id is None
                            else concurrent.members(str(workload_run_id))
                        ),
                        gpu_samples=gpu_samples,
                    )
                )
        return tuple(observations)


def _load_placement_context(
    connection: sqlite3.Connection,
    placement_id: str,
) -> _PlacementContext:
    placements = DeploymentPlacementRepository(connection)
    placement = placements.get(placement_id)
    if placement is None:
        raise DeploymentAnalysisError(
            f"deployment placement not found: {placement_id}"
        )
    deployment = DeploymentCandidateRepository(connection).get(
        placement.deployment_candidate_id
    )
    if deployment is None:
        raise DeploymentAnalysisError(
            "deployment candidate missing for persisted placement"
        )
    candidates_repo = CandidateRepository(connection)
    resolved_repo = PlacementRepository(connection)
    candidates: dict[str, Candidate] = {}
    resolved: dict[str, ResolvedPlacementRecord] = {}
    backends: dict[str, tuple[str, ...]] = {}
    resolved_ids = {
        item.instance_id: item.resolved_placement_id
        for item in placement.instance_placements
    }
    for instance in deployment.instances:
        candidate = candidates_repo.get(instance.candidate_id)
        if candidate is None:
            raise DeploymentAnalysisError(
                f"candidate missing for deployment instance: "
                f"{instance.instance_id}"
            )
        placement_record = resolved_repo.get(
            resolved_ids[instance.instance_id]
        )
        if placement_record is None:
            raise DeploymentAnalysisError(
                f"resolved placement missing for instance: "
                f"{instance.instance_id}"
            )
        candidates[instance.instance_id] = candidate
        resolved[instance.instance_id] = placement_record
        backend_rows = connection.execute(
            """
            SELECT logical_device_name, backend
            FROM accelerator_device
            WHERE host_id = ? AND binary_id = ?
            ORDER BY logical_device_name
            """,
            (placement.host_id, instance.binary_id),
        ).fetchall()
        backend_by_device = {
            str(row["logical_device_name"]): str(row["backend"])
            for row in backend_rows
        }
        devices = placement_record.devices
        if devices == "auto":
            backends[instance.instance_id] = tuple(
                backend_by_device[name]
                for name in sorted(backend_by_device)
            )
        elif isinstance(devices, tuple):
            backends[instance.instance_id] = tuple(
                backend_by_device[name]
                for name in devices
                if name in backend_by_device
            )
        else:
            raise DeploymentAnalysisError(
                "persisted resolved placement devices are invalid"
            )
    return _PlacementContext(
        placement_id=placement_id,
        deployment_candidate_id=placement.deployment_candidate_id,
        deployment=deployment,
        candidates=candidates,
        resolved=resolved,
        backends=backends,
        memory=placements.memory(placement_id),
        allocations=placements.allocations(placement_id),
    )


def _valid(item: _Observation) -> bool:
    return (
        item.deployment_status == "completed"
        and item.workload_run_id is not None
        and item.workload_status == "completed"
        and item.correctness_valid is True
    )


def _matches_filters(
    item: _Observation,
    filters: Sequence[DeploymentAnalysisFilter],
    *,
    missing_is_false: bool = False,
) -> bool:
    for condition in filters:
        try:
            value = _resolve_path(item, condition.path)
        except DeploymentAnalysisError:
            if missing_is_false:
                return False
            raise
        if value != condition.value:
            return False
    return True


def _resolve_path(item: _Observation, path: str | None) -> JsonScalar:
    if path is None:
        return None
    fixed: dict[str, JsonScalar] = {
        "deployment.candidate_id": item.context.deployment_candidate_id,
        "deployment.placement_id": item.context.placement_id,
        "run.id": item.deployment_run_id,
        "run.status": item.deployment_status,
        "run.quality": item.deployment_quality,
        "run.failure_kind": item.deployment_failure_kind,
        "workload.run_id": item.workload_run_id,
        "workload.case_id": item.workload_case_id,
        "workload.phase": item.phase,
        "workload.status": item.workload_status,
        "workload.quality": item.workload_quality,
        "workload.correctness_valid": item.correctness_valid,
    }
    if path in fixed:
        return fixed[path]

    if path.startswith("instance."):
        parts = path.split(".")
        if len(parts) < 3:
            raise DeploymentAnalysisError(
                f"unknown deployment analysis path: {path}"
            )
        instance_id = parts[1]
        instance = next(
            (
                value
                for value in item.context.deployment.instances
                if value.instance_id == instance_id
            ),
            None,
        )
        if instance is None:
            raise DeploymentAnalysisError(
                f"unknown deployment instance in path: {path}"
            )
        root = parts[2]
        if root == "candidate":
            payload: Any = item.context.candidates[
                instance_id
            ].model_dump(mode="python", by_alias=True)
            return _nested_scalar(payload, parts[3:], path)
        if root == "resolved":
            payload = {
                "production_context_size": (
                    item.context.resolved[
                        instance_id
                    ].production_context_size
                ),
                "n_gpu_layers": item.context.resolved[
                    instance_id
                ].n_gpu_layers,
                "n_cpu_moe": item.context.resolved[
                    instance_id
                ].n_cpu_moe,
                "split_mode": item.context.resolved[
                    instance_id
                ].split_mode,
                "main_gpu": item.context.resolved[
                    instance_id
                ].main_gpu,
                "devices": item.context.resolved[instance_id].devices,
                "tensor_split": item.context.resolved[
                    instance_id
                ].tensor_split,
                "backends": item.context.backends.get(instance_id, ()),
            }
            return _nested_scalar(payload, parts[3:], path)
        if root == "binary_id" and len(parts) == 3:
            return instance.binary_id
        if root == "requested_placement":
            payload = instance.requested_placement.model_dump(
                mode="python",
                by_alias=True,
            )
            return _nested_scalar(payload, parts[3:], path)

    if path.startswith("workload.member."):
        if item.workload is None:
            raise DeploymentAnalysisError(
                f"workload coordinate unavailable for path: {path}"
            )
        parts = path.split(".")
        if len(parts) < 4:
            raise DeploymentAnalysisError(
                f"unknown workload member path: {path}"
            )
        instance_id = parts[2]
        member = next(
            (
                value
                for value in item.workload.members
                if value.instance_id == instance_id
            ),
            None,
        )
        if member is None:
            raise DeploymentAnalysisError(
                f"unknown workload member in path: {path}"
            )
        return _nested_scalar(
            member.model_dump(mode="python"),
            parts[3:],
            path,
        )

    if path.startswith("placement.device."):
        parts = path.split(".")
        if len(parts) != 4:
            raise DeploymentAnalysisError(
                f"unknown placement device path: {path}"
            )
        device_id = parts[2]
        allocation = next(
            (
                value
                for value in item.context.allocations
                if value.device_id == device_id
            ),
            None,
        )
        if allocation is None:
            raise DeploymentAnalysisError(
                f"unknown placement device in path: {path}"
            )
        field = parts[3]
        if field not in {
            "projected_bytes",
            "reserved_margin_bytes",
            "device_total_bytes",
            "projected_free_bytes",
        }:
            raise DeploymentAnalysisError(
                f"unknown placement device path: {path}"
            )
        return int(getattr(allocation, field))

    raise DeploymentAnalysisError(
        f"unknown deployment analysis path: {path}"
    )


def _nested_scalar(
    value: Any,
    segments: Sequence[str],
    path: str,
) -> JsonScalar:
    current = value
    for segment in segments:
        if isinstance(current, Mapping):
            if segment not in current:
                raise DeploymentAnalysisError(
                    f"unknown deployment analysis path: {path}"
                )
            current = current[segment]
            continue
        if isinstance(current, (tuple, list)):
            try:
                index = int(segment)
                current = current[index]
            except (ValueError, IndexError) as exc:
                raise DeploymentAnalysisError(
                    f"unknown deployment analysis path: {path}"
                ) from exc
            continue
        raise DeploymentAnalysisError(
            f"unknown deployment analysis path: {path}"
        )
    if current is not None and not isinstance(
        current,
        (str, int, float, bool),
    ):
        raise DeploymentAnalysisError(
            f"deployment analysis path is not scalar: {path}"
        )
    return current


def _evaluate_metric(
    metric: str,
    observations: Sequence[_Observation],
) -> float | None:
    if not observations:
        return None
    if _workload_sensitive_metric(metric):
        signatures = {_workload_signature(item) for item in observations}
        if len(signatures) != 1:
            raise DeploymentAnalysisError(
                f"metric {metric!r} matches multiple workload coordinates; "
                "add exact workload filters"
            )

    if metric == "deployment.combined_tg_tps":
        return _mean(
            item.combined_decode_tps
            for item in observations
            if item.combined_decode_tps is not None
        )
    if metric == "deployment.combined_pp_tps":
        return _mean(
            item.combined_prompt_tps
            for item in observations
            if item.combined_prompt_tps is not None
        )
    if metric == "deployment.min_retention":
        values = [
            item.min_retention
            for item in observations
            if item.min_retention is not None
        ]
        return min(values) if values else None

    context = observations[0].context
    if metric == "deployment.total_validated_context_tokens":
        return float(
            sum(
                item.production_context_size
                for item in context.resolved.values()
            )
        )

    projected = {
        item.device_id: item.projected_free_bytes
        for item in context.allocations
    }
    runtime = _combined_runtime_summary(observations)

    if metric == "deployment.min_device_projected_headroom_bytes":
        return float(min(projected.values())) if projected else None
    if metric == "deployment.min_device_runtime_headroom_bytes":
        values = list(runtime.min_free_by_device.values())
        return float(min(values)) if values else None
    if metric == "deployment.min_device_headroom_bytes":
        if (
            not projected
            or set(runtime.min_free_by_device) != set(projected)
        ):
            return None
        return float(min(runtime.min_free_by_device.values()))
    if metric == "deployment.total_power_avg_w":
        return runtime.total_power_avg_w
    if metric == "deployment.total_power_peak_w":
        return runtime.total_power_peak_w

    prefix = "deployment.device."
    if metric.startswith(prefix):
        remainder = metric.removeprefix(prefix)
        device_id, separator, field = remainder.rpartition(".")
        if not separator or not device_id:
            raise DeploymentAnalysisError(
                f"unknown deployment metric: {metric}"
            )
        if field == "projected_headroom_bytes":
            value = projected.get(device_id)
            return None if value is None else float(value)
        if field == "runtime_headroom_bytes":
            value = runtime.min_free_by_device.get(device_id)
            return None if value is None else float(value)
        if field == "headroom_bytes":
            value = runtime.min_free_by_device.get(device_id)
            return None if value is None else float(value)

    instance_prefix = "deployment.instance."
    if metric.startswith(instance_prefix):
        remainder = metric.removeprefix(instance_prefix)
        instance_id, separator, field = remainder.rpartition(".")
        if not separator or not instance_id:
            raise DeploymentAnalysisError(
                f"unknown deployment metric: {metric}"
            )
        members = [
            member
            for item in observations
            for member in item.members
            if member.instance_id == instance_id
        ]
        if field == "pp_tps":
            return _mean(
                member.native_prompt_tps
                for member in members
                if member.native_prompt_tps is not None
            )
        if field == "tg_tps":
            return _mean(
                member.native_decode_tps
                for member in members
                if member.native_decode_tps is not None
            )
        if field == "retention":
            values = [
                member.retention
                for member in members
                if member.retention is not None
            ]
            return min(values) if values else None
        if field == "latency_ms":
            return _mean(
                member.latency_ms
                for member in members
                if member.latency_ms is not None
            )
        if field == "latency_increase_pct":
            return _mean(
                member.latency_increase_pct
                for member in members
                if member.latency_increase_pct is not None
            )

    raise DeploymentAnalysisError(f"unknown deployment metric: {metric}")


def _workload_sensitive_metric(metric: str) -> bool:
    return (
        metric
        in {
            "deployment.combined_tg_tps",
            "deployment.combined_pp_tps",
        }
        or metric.startswith("deployment.instance.")
    )


def _workload_signature(item: _Observation) -> tuple[Any, ...]:
    if item.workload is None:
        return (None,)
    return (
        item.phase,
        tuple(
            (
                member.instance_id,
                member.mode,
                int(member.prompt_tokens),
                int(member.generate_tokens),
                int(member.depth_tokens),
            )
            for member in item.workload.members
        ),
    )


def _runtime_summary(item: _Observation) -> _RuntimeSummary:
    allocations = {
        value.device_id: value
        for value in item.context.allocations
    }
    peak: dict[str, int] = {}
    minimum_free: dict[str, int] = {}
    total_power: list[float] = []
    planned = set(allocations)

    for _, samples in item.gpu_samples:
        by_key = {
            sample.stable_device_key: sample
            for sample in samples
            if sample.stable_device_key is not None
        }
        powers: list[float] = []
        complete_power = bool(planned)
        for device_id in planned:
            sample = by_key.get(device_id)
            if sample is None:
                complete_power = False
                continue
            if sample.vram_used_bytes is not None:
                peak[device_id] = max(
                    peak.get(device_id, 0),
                    int(sample.vram_used_bytes),
                )
            if (
                sample.vram_total_bytes is not None
                and sample.vram_used_bytes is not None
            ):
                free = max(
                    0,
                    int(sample.vram_total_bytes)
                    - int(sample.vram_used_bytes),
                )
                prior = minimum_free.get(device_id)
                minimum_free[device_id] = (
                    free if prior is None else min(prior, free)
                )
            if sample.power_w is None:
                complete_power = False
            else:
                powers.append(float(sample.power_w))
        if complete_power and len(powers) == len(planned):
            total_power.append(sum(powers))

    return _RuntimeSummary(
        peak_used_by_device=peak,
        min_free_by_device=minimum_free,
        total_power_avg_w=(
            fmean(total_power) if total_power else None
        ),
        total_power_peak_w=(
            max(total_power) if total_power else None
        ),
        total_power_sum_w=sum(total_power),
        total_power_sample_count=len(total_power),
    )


def _combined_runtime_summary(
    observations: Sequence[_Observation],
) -> _RuntimeSummary:
    unique: dict[str, _Observation] = {}
    for item in observations:
        unique[item.deployment_run_id] = item
    summaries = [_runtime_summary(item) for item in unique.values()]
    peak: dict[str, int] = {}
    minimum_free: dict[str, int] = {}
    power_sum = 0.0
    power_count = 0
    peaks: list[float] = []
    for summary in summaries:
        for device_id, value in summary.peak_used_by_device.items():
            peak[device_id] = max(peak.get(device_id, 0), value)
        for device_id, value in summary.min_free_by_device.items():
            prior = minimum_free.get(device_id)
            minimum_free[device_id] = (
                value if prior is None else min(prior, value)
            )
        power_sum += summary.total_power_sum_w
        power_count += summary.total_power_sample_count
        if summary.total_power_peak_w is not None:
            peaks.append(summary.total_power_peak_w)
    return _RuntimeSummary(
        peak_used_by_device=peak,
        min_free_by_device=minimum_free,
        total_power_avg_w=(
            power_sum / power_count
            if power_count > 0
            else None
        ),
        total_power_peak_w=max(peaks) if peaks else None,
        total_power_sum_w=power_sum,
        total_power_sample_count=power_count,
    )


def _constraint_failure(
    observations: Sequence[_Observation],
    constraints: Sequence[DeploymentMetricConstraint],
) -> str | None:
    for constraint in constraints:
        selected = [
            item
            for item in observations
            if _matches_filters(
                item,
                constraint.filters,
                missing_is_false=True,
            )
        ]
        value = _evaluate_metric(constraint.metric, selected)
        if value is None:
            return f"constraint metric unavailable: {constraint.metric}"
        if not _compare(value, constraint.operator, constraint.value):
            return (
                f"constraint failed: {constraint.metric} "
                f"{constraint.operator} {constraint.value}"
            )
    return None


def _compare(left: float, operator: str, right: float) -> bool:
    if operator == "ge":
        return left >= right
    if operator == "gt":
        return left > right
    if operator == "le":
        return left <= right
    if operator == "lt":
        return left < right
    if operator == "eq":
        return left == right
    raise AssertionError(f"unsupported constraint operator: {operator}")


def _dominates(
    left: DeploymentParetoPoint,
    right: DeploymentParetoPoint,
    objectives: Sequence[DeploymentParetoObjective],
) -> bool:
    no_worse = True
    strictly_better = False
    for objective in objectives:
        left_value = left.values[objective.key]
        right_value = right.values[objective.key]
        if objective.direction == "maximize":
            if left_value < right_value:
                no_worse = False
                break
            strictly_better = strictly_better or left_value > right_value
        else:
            if left_value > right_value:
                no_worse = False
                break
            strictly_better = strictly_better or left_value < right_value
    return no_worse and strictly_better


def _export_row(item: _Observation) -> dict[str, Any]:
    row: dict[str, Any] = {
        "deployment_candidate_id": item.context.deployment_candidate_id,
        "deployment_placement_id": item.context.placement_id,
        "deployment_run_id": item.deployment_run_id,
        "deployment_status": item.deployment_status,
        "deployment_quality": item.deployment_quality,
        "deployment_failure_kind": item.deployment_failure_kind,
        "workload_run_id": item.workload_run_id,
        "workload_case_id": item.workload_case_id,
        "phase": item.phase,
        "workload_status": item.workload_status,
        "workload_quality": item.workload_quality,
        "correctness_valid": item.correctness_valid,
        "combined_pp_tps": item.combined_prompt_tps,
        "combined_tg_tps": item.combined_decode_tps,
        "min_retention": item.min_retention,
        "total_validated_context_tokens": _evaluate_metric(
            "deployment.total_validated_context_tokens",
            (item,),
        ),
        "min_device_projected_headroom_bytes": _evaluate_metric(
            "deployment.min_device_projected_headroom_bytes",
            (item,),
        ),
        "min_device_runtime_headroom_bytes": _evaluate_metric(
            "deployment.min_device_runtime_headroom_bytes",
            (item,),
        ),
        "total_power_avg_w": _evaluate_metric(
            "deployment.total_power_avg_w",
            (item,),
        ),
        "total_power_peak_w": _evaluate_metric(
            "deployment.total_power_peak_w",
            (item,),
        ),
    }
    for instance_id, candidate in sorted(
        item.context.candidates.items()
    ):
        prefix = f"instance.{instance_id}."
        resolved = item.context.resolved[instance_id]
        row[prefix + "context_tokens"] = resolved.production_context_size
        row[prefix + "cache_type_k"] = candidate.context.cache_type_k
        row[prefix + "cache_type_v"] = candidate.context.cache_type_v
        row[prefix + "batch_size"] = candidate.compute.batch_size
        row[prefix + "ubatch_size"] = candidate.compute.ubatch_size
        row[prefix + "n_gpu_layers"] = resolved.n_gpu_layers
        row[prefix + "split_mode"] = resolved.split_mode
        row[prefix + "tensor_split_json"] = json.dumps(
            resolved.tensor_split,
            separators=(",", ":"),
        )
        member = next(
            (
                value
                for value in item.members
                if value.instance_id == instance_id
            ),
            None,
        )
        if member is not None:
            row[prefix + "mode"] = member.mode
            row[prefix + "native_tps"] = (
                member.native_prompt_tps
                if member.mode == "prefill"
                else member.native_decode_tps
            )
            row[prefix + "retention"] = member.retention
            row[prefix + "latency_ms"] = member.latency_ms
            row[prefix + "latency_increase_pct"] = (
                member.latency_increase_pct
            )
    return row


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def _mean(values: Iterable[int | float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return fmean(finite) if finite else None


def _sorted_scalars(values: set[JsonScalar]) -> tuple[JsonScalar, ...]:
    return tuple(sorted(values, key=_scalar_sort_key))


def _scalar_sort_key(value: JsonScalar) -> tuple[int, float | str]:
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, 1.0 if value else 0.0)
    if isinstance(value, (int, float)):
        return (2, float(value))
    return (3, value)
