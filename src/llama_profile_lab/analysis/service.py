"""Read-only multidimensional analysis over persisted benchmark observations."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from llama_profile_lab.analysis.metrics import (
    DEFAULT_METRIC_REGISTRY,
    MetricRegistry,
)
from llama_profile_lab.analysis.models import (
    AnalysisFilter,
    BaselineDelta,
    CandidateComparison,
    CurvePoint,
    LatencyEstimate,
    MatrixCell,
    MatrixFacet,
    MatrixProjection,
    ParetoCandidate,
    ParetoObjective,
    ParetoResult,
)
from llama_profile_lab.db import (
    Database,
    ExperimentRepository,
    SearchSpaceRepository,
)
from llama_profile_lab.domain import Candidate
from llama_profile_lab.domain.base import JsonScalar
from llama_profile_lab.domain.workload import WorkloadCase

_WORKLOAD_ADAPTER: TypeAdapter[WorkloadCase] = TypeAdapter(WorkloadCase)


class AnalysisError(RuntimeError):
    """Raised when requested analysis is missing data or is ambiguous."""


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    """One completed benchmark attempt enriched with immutable coordinates."""

    run_id: str
    candidate_id: str
    candidate_ordinal: int
    candidate: Candidate
    workload_case_id: str
    suite_case_index: int
    workload: WorkloadCase
    host_id: str
    binary_id: str
    quality: str | None
    duration_ns: int | None
    samples: tuple[float, ...]
    metrics: Mapping[str, int | float]

    @property
    def token_count(self) -> int:
        prompt = int(getattr(self.workload, "prompt_tokens", 0))
        generated = int(getattr(self.workload, "generate_tokens", 0))
        return prompt + generated


@dataclass(frozen=True, slots=True)
class AnalysisDataset:
    """Completed observation set and experiment metadata."""

    experiment_id: str
    base_candidate_id: str
    baseline_candidate_id: str
    dimension_paths: tuple[str, ...]
    candidate_ordinals: Mapping[str, int]
    runs: tuple[AnalysisRun, ...]


class AnalysisService:
    """Stateless analysis services backed by the canonical SQLite database."""

    def __init__(
        self,
        database: Database,
        *,
        metrics: MetricRegistry = DEFAULT_METRIC_REGISTRY,
    ) -> None:
        self.database = database
        self.metrics = metrics

    def matrix(
        self,
        experiment_id: str,
        *,
        x_path: str,
        y_path: str,
        metric: str,
        filters: Sequence[AnalysisFilter] = (),
        facet_path: str | None = None,
        qualities: Sequence[str] = (),
    ) -> MatrixProjection:
        """Project one unambiguous Candidate/workload slice into sparse X/Y cells."""
        dataset = self._load(experiment_id)
        selected = _select_runs(dataset.runs, filters, qualities)
        if not selected:
            raise AnalysisError("no completed runs match the matrix filters")

        groups: dict[tuple[JsonScalar, JsonScalar, JsonScalar], list[AnalysisRun]] = (
            defaultdict(list)
        )
        for run in selected:
            x_value = _resolve_path(run, x_path)
            y_value = _resolve_path(run, y_path)
            facet_value = _resolve_path(run, facet_path) if facet_path else None
            groups[(facet_value, x_value, y_value)].append(run)

        facets: dict[JsonScalar, list[MatrixCell]] = defaultdict(list)
        for (facet_value, x_value, y_value), runs in groups.items():
            candidate_ids = {run.candidate_id for run in runs}
            suite_indices = {run.suite_case_index for run in runs}
            if len(candidate_ids) != 1 or len(suite_indices) != 1:
                raise AnalysisError(
                    "matrix cell collapses multiple Candidate/workload coordinates; "
                    "add filters or a facet for remaining dimensions"
                )
            value = self.metrics.evaluate(metric, runs)
            if value is None:
                continue
            workload_ids = {run.workload_case_id for run in runs}
            if len(workload_ids) != 1:
                raise AnalysisError(
                    "matrix cell maps to multiple concrete workloads; add a workload filter"
                )
            facets[facet_value].append(
                MatrixCell(
                    x=x_value,
                    y=y_value,
                    value=value,
                    candidate_id=next(iter(candidate_ids)),
                    workload_case_id=next(iter(workload_ids)),
                    run_count=len(runs),
                    sample_count=sum(len(run.samples) for run in runs),
                )
            )

        if not facets:
            raise AnalysisError(f"metric {metric!r} has no values for the selected runs")

        x_values = _sorted_scalars({cell.x for cells in facets.values() for cell in cells})
        y_values = _sorted_scalars({cell.y for cells in facets.values() for cell in cells})
        facet_models = tuple(
            MatrixFacet(
                value=facet_value,
                cells=tuple(
                    sorted(
                        cells,
                        key=lambda cell: (
                            _scalar_sort_key(cell.y),
                            _scalar_sort_key(cell.x),
                        ),
                    )
                ),
            )
            for facet_value, cells in sorted(
                facets.items(),
                key=lambda item: _scalar_sort_key(item[0]),
            )
        )
        return MatrixProjection(
            experiment_id=experiment_id,
            x_path=x_path,
            y_path=y_path,
            metric=metric,
            facet_path=facet_path,
            x_values=x_values,
            y_values=y_values,
            facets=facet_models,
        )

    def compare(
        self,
        experiment_id: str,
        *,
        candidate_id: str,
        metric_names: Sequence[str],
        filters: Sequence[AnalysisFilter] = (),
        qualities: Sequence[str] = (),
        baseline_candidate_id: str | None = None,
    ) -> CandidateComparison:
        """Compare signed metric deltas against the configured or explicit baseline."""
        dataset = self._load(experiment_id)
        baseline_id = baseline_candidate_id or dataset.baseline_candidate_id
        selected = _select_runs(dataset.runs, filters, qualities)
        candidate_runs = [run for run in selected if run.candidate_id == candidate_id]
        baseline_runs = [run for run in selected if run.candidate_id == baseline_id]
        if not candidate_runs:
            raise AnalysisError(f"candidate has no matching completed runs: {candidate_id}")
        if not baseline_runs:
            raise AnalysisError(f"baseline has no matching completed runs: {baseline_id}")

        candidate_groups = _group_by_suite_case(candidate_runs)
        baseline_groups = _group_by_suite_case(baseline_runs)
        common = sorted(set(candidate_groups) & set(baseline_groups))
        if not common:
            raise AnalysisError("candidate and baseline have no matching suite workloads")

        deltas: list[BaselineDelta] = []
        for suite_case_index in common:
            current = candidate_groups[suite_case_index]
            baseline = baseline_groups[suite_case_index]
            label = _workload_label(current[0])
            for metric_name in metric_names:
                current_value = self.metrics.evaluate(metric_name, current)
                baseline_value = self.metrics.evaluate(metric_name, baseline)
                delta: float | None = None
                percent: float | None = None
                if current_value is not None and baseline_value is not None:
                    delta = current_value - baseline_value
                    if baseline_value != 0:
                        percent = delta / abs(baseline_value) * 100.0
                deltas.append(
                    BaselineDelta(
                        suite_case_index=suite_case_index,
                        workload_label=label,
                        metric=metric_name,
                        baseline_value=baseline_value,
                        candidate_value=current_value,
                        delta=delta,
                        percent_delta=percent,
                    )
                )
        return CandidateComparison(
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            baseline_candidate_id=baseline_id,
            deltas=tuple(deltas),
        )

    def pareto(
        self,
        experiment_id: str,
        *,
        objectives: Sequence[ParetoObjective],
        filters: Sequence[AnalysisFilter] = (),
        qualities: Sequence[str] = (),
    ) -> ParetoResult:
        """Return the non-dominated Candidate set for explicit user objectives."""
        if not objectives:
            raise AnalysisError("at least one Pareto objective is required")
        keys = [objective.key for objective in objectives]
        if len(keys) != len(set(keys)):
            raise AnalysisError("Pareto objective keys must be unique")

        dataset = self._load(experiment_id)
        selected = _select_runs(dataset.runs, filters, qualities)
        candidate_ordinals = dataset.candidate_ordinals
        vectors: list[ParetoCandidate] = []
        excluded: dict[str, str] = {}

        for candidate_id in sorted(
            candidate_ordinals,
            key=lambda item: (candidate_ordinals[item], item),
        ):
            values: dict[str, float] = {}
            reason: str | None = None
            for objective in objectives:
                objective_runs = [
                    run
                    for run in selected
                    if run.candidate_id == candidate_id
                    and _matches_filters(run, objective.filters)
                ]
                if not objective_runs:
                    reason = f"no data for objective {objective.key}"
                    break
                suite_indices = {run.suite_case_index for run in objective_runs}
                if len(suite_indices) != 1:
                    raise AnalysisError(
                        f"objective {objective.key!r} matches multiple suite workloads; "
                        "add workload filters"
                    )
                value = self.metrics.evaluate(objective.metric, objective_runs)
                if value is None:
                    reason = f"metric {objective.metric} unavailable for {objective.key}"
                    break
                values[objective.key] = value
            if reason is not None:
                excluded[candidate_id] = reason
                continue
            vectors.append(
                ParetoCandidate(
                    candidate_id=candidate_id,
                    candidate_ordinal=candidate_ordinals[candidate_id],
                    values=values,
                )
            )

        frontier = tuple(
            candidate
            for candidate in vectors
            if not any(
                _dominates(other, candidate, objectives)
                for other in vectors
                if other.candidate_id != candidate.candidate_id
            )
        )
        return ParetoResult(
            experiment_id=experiment_id,
            objectives=tuple(objectives),
            evaluated_count=len(vectors),
            frontier=frontier,
            excluded=excluded,
        )

    def latency(
        self,
        experiment_id: str,
        *,
        candidate_id: str,
        prompt_tokens: int,
        generate_tokens: int,
        decode_start_depth_tokens: int | None = None,
        filters: Sequence[AnalysisFilter] = (),
        qualities: Sequence[str] = (),
    ) -> LatencyEstimate:
        """Estimate compute latency using linearly interpolated PP/TG throughput curves."""
        if prompt_tokens <= 0 or generate_tokens <= 0:
            raise AnalysisError("prompt and generate token counts must be positive")
        dataset = self._load(experiment_id)
        selected = [
            run
            for run in _select_runs(dataset.runs, filters, qualities)
            if run.candidate_id == candidate_id
        ]
        if not selected:
            raise AnalysisError(f"candidate has no matching completed runs: {candidate_id}")

        candidate = selected[0].candidate
        start_depth = (
            prompt_tokens if decode_start_depth_tokens is None else decode_start_depth_tokens
        )
        if start_depth < 0:
            raise AnalysisError("decode start depth must be non-negative")
        if start_depth + generate_tokens > candidate.context.size:
            raise AnalysisError("requested decode interval exceeds Candidate context size")

        prefill = [
            run
            for run in selected
            if getattr(run.workload, "kind", "") == "microbench-prefill"
            and int(getattr(run.workload, "depth_tokens", -1)) == 0
        ]
        decode = [
            run
            for run in selected
            if getattr(run.workload, "kind", "") == "microbench-decode"
        ]
        prefill_curve = self._prefill_curve(prefill)
        decode_curve = self._decode_curve(decode)
        if not prefill_curve:
            raise AnalysisError("latency model requires d0 prefill benchmark data")
        if not decode_curve:
            raise AnalysisError("latency model requires decode benchmark data")

        prefill_tps = _interpolate(prefill_curve, prompt_tokens)
        prefill_seconds = prompt_tokens / prefill_tps
        decode_seconds = sum(
            1.0 / _interpolate(decode_curve, start_depth + offset)
            for offset in range(generate_tokens)
        )
        decode_average = generate_tokens / decode_seconds
        return LatencyEstimate(
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            prompt_tokens=prompt_tokens,
            generate_tokens=generate_tokens,
            decode_start_depth_tokens=start_depth,
            prefill_tokens_per_second=prefill_tps,
            decode_average_tokens_per_second=decode_average,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            total_seconds=prefill_seconds + decode_seconds,
            prefill_curve=tuple(
                CurvePoint(tokens=tokens, tokens_per_second=value)
                for tokens, value in prefill_curve
            ),
            decode_curve=tuple(
                CurvePoint(tokens=tokens, tokens_per_second=value)
                for tokens, value in decode_curve
            ),
        )

    def export_rows(
        self,
        experiment_id: str,
        *,
        filters: Sequence[AnalysisFilter] = (),
        qualities: Sequence[str] = (),
        metric_names: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Export one summarized row per Candidate × suite workload."""
        dataset = self._load(experiment_id)
        selected = _select_runs(dataset.runs, filters, qualities)
        grouped: dict[tuple[int, int], list[AnalysisRun]] = defaultdict(list)
        for run in selected:
            grouped[(run.candidate_ordinal, run.suite_case_index)].append(run)

        metrics = tuple(metric_names or _default_export_metrics())
        rows: list[dict[str, Any]] = []
        for key in sorted(grouped):
            runs = grouped[key]
            first = runs[0]
            row: dict[str, Any] = {
                "experiment_id": experiment_id,
                "candidate_id": first.candidate_id,
                "candidate_ordinal": first.candidate_ordinal,
                "suite_case_index": first.suite_case_index,
                "workload_case_id": first.workload_case_id,
                "workload_kind": getattr(first.workload, "kind", ""),
                "prompt_tokens": getattr(first.workload, "prompt_tokens", None),
                "generate_tokens": getattr(first.workload, "generate_tokens", None),
                "depth_tokens": getattr(first.workload, "depth_tokens", None),
                "run_count": len(runs),
                "qualities": sorted(
                    {run.quality or "unclassified" for run in runs}
                ),
            }
            for path in dataset.dimension_paths:
                row[f"candidate.{path}"] = _resolve_path(first, path)
            for metric_name in metrics:
                row[metric_name] = self.metrics.evaluate(metric_name, runs)
            rows.append(row)
        return tuple(rows)

    def _prefill_curve(
        self,
        runs: Sequence[AnalysisRun],
    ) -> tuple[tuple[int, float], ...]:
        grouped: dict[int, list[AnalysisRun]] = defaultdict(list)
        for run in runs:
            grouped[int(getattr(run.workload, "prompt_tokens", 0))].append(run)
        points = []
        for prompt_tokens, point_runs in grouped.items():
            value = self.metrics.evaluate("throughput.median", point_runs)
            if value is not None and value > 0:
                points.append((prompt_tokens, value))
        return tuple(sorted(points))

    def _decode_curve(
        self,
        runs: Sequence[AnalysisRun],
    ) -> tuple[tuple[int, float], ...]:
        grouped: dict[int, list[AnalysisRun]] = defaultdict(list)
        for run in runs:
            grouped[int(getattr(run.workload, "depth_tokens", 0))].append(run)
        points = []
        for depth_tokens, point_runs in grouped.items():
            generate_lengths = {
                int(getattr(run.workload, "generate_tokens", 0)) for run in point_runs
            }
            if len(generate_lengths) != 1:
                raise AnalysisError(
                    "decode curve has multiple generation lengths at the same depth; "
                    "add workload filters"
                )
            value = self.metrics.evaluate("throughput.median", point_runs)
            if value is not None and value > 0:
                points.append((depth_tokens, value))
        return tuple(sorted(points))

    def _load(self, experiment_id: str) -> AnalysisDataset:
        with self.database.session() as connection:
            return _load_dataset(connection, experiment_id)


def serialize_export(
    rows: Sequence[Mapping[str, Any]],
    *,
    format_name: str,
) -> str:
    """Serialize summarized rows as deterministic JSON or CSV."""
    if format_name == "json":
        return json.dumps(list(rows), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if format_name != "csv":
        raise AnalysisError(f"unsupported export format: {format_name}")
    if not rows:
        return ""
    fieldnames = _export_fieldnames(rows)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                name: _csv_value(row.get(name))
                for name in fieldnames
            }
        )
    return output.getvalue()


def _load_dataset(
    connection: sqlite3.Connection,
    experiment_id: str,
) -> AnalysisDataset:
    experiments = ExperimentRepository(connection)
    record = experiments.get(experiment_id)
    definition = experiments.get_definition(experiment_id)
    if record is None or definition is None:
        raise AnalysisError(f"experiment not found: {experiment_id}")

    search_space = SearchSpaceRepository(connection).get(definition.search_space_id)
    if search_space is None:
        raise AnalysisError("experiment SearchSpace is missing")
    if definition.baseline.type == "base-candidate":
        baseline_id = definition.base_candidate_id
    else:
        baseline_id = definition.baseline.candidate_id

    candidate_rows = connection.execute(
        """
        SELECT candidate_id, ordinal
        FROM experiment_candidate
        WHERE experiment_id = ?
        ORDER BY ordinal, candidate_id
        """,
        (experiment_id,),
    ).fetchall()
    candidate_ordinals = {
        str(row["candidate_id"]): int(row["ordinal"])
        for row in candidate_rows
    }

    rows = connection.execute(
        """
        SELECT br.id AS run_id, br.host_id, br.binary_id, br.quality,
               br.duration_ns, bc.candidate_id, bc.workload_case_id,
               ec.ordinal AS candidate_ordinal, ew.suite_case_index,
               c.config_json, wc.definition_json
        FROM benchmark_run AS br
        JOIN benchmark_case AS bc ON bc.id = br.benchmark_case_id
        JOIN experiment_candidate AS ec
          ON ec.experiment_id = bc.experiment_id
         AND ec.candidate_id = bc.candidate_id
        JOIN experiment_workload AS ew
          ON ew.experiment_id = bc.experiment_id
         AND ew.candidate_id = bc.candidate_id
         AND ew.workload_case_id = bc.workload_case_id
        JOIN candidate AS c ON c.id = bc.candidate_id
        JOIN workload_case AS wc ON wc.id = bc.workload_case_id
        WHERE bc.experiment_id = ? AND br.status = 'completed'
        ORDER BY ec.ordinal, ew.suite_case_index, br.started_at, br.id
        """,
        (experiment_id,),
    ).fetchall()

    sample_rows = connection.execute(
        """
        SELECT bs.run_id, bs.tokens_per_second
        FROM benchmark_sample AS bs
        JOIN benchmark_run AS br ON br.id = bs.run_id
        JOIN benchmark_case AS bc ON bc.id = br.benchmark_case_id
        WHERE bc.experiment_id = ? AND br.status = 'completed'
        ORDER BY bs.run_id, bs.sample_index
        """,
        (experiment_id,),
    ).fetchall()
    samples: dict[str, list[float]] = defaultdict(list)
    for row in sample_rows:
        samples[str(row["run_id"])].append(float(row["tokens_per_second"]))

    metric_rows = connection.execute(
        """
        SELECT m.run_id, m.metric_name, m.value_real, m.value_integer
        FROM metric AS m
        JOIN benchmark_run AS br ON br.id = m.run_id
        JOIN benchmark_case AS bc ON bc.id = br.benchmark_case_id
        WHERE bc.experiment_id = ? AND br.status = 'completed'
        ORDER BY m.run_id, m.metric_name
        """,
        (experiment_id,),
    ).fetchall()
    metrics: dict[str, dict[str, int | float]] = defaultdict(dict)
    for row in metric_rows:
        run_metrics = metrics[str(row["run_id"])]
        if row["value_integer"] is not None:
            run_metrics[str(row["metric_name"])] = int(row["value_integer"])
        elif row["value_real"] is not None:
            run_metrics[str(row["metric_name"])] = float(row["value_real"])

    analysis_runs = tuple(
        AnalysisRun(
            run_id=str(row["run_id"]),
            candidate_id=str(row["candidate_id"]),
            candidate_ordinal=int(row["candidate_ordinal"]),
            candidate=Candidate.model_validate_json(str(row["config_json"])),
            workload_case_id=str(row["workload_case_id"]),
            suite_case_index=int(row["suite_case_index"]),
            workload=_WORKLOAD_ADAPTER.validate_json(str(row["definition_json"])),
            host_id=str(row["host_id"]),
            binary_id=str(row["binary_id"]),
            quality=row["quality"],
            duration_ns=row["duration_ns"],
            samples=tuple(samples.get(str(row["run_id"]), ())),
            metrics=metrics.get(str(row["run_id"]), {}),
        )
        for row in rows
    )
    return AnalysisDataset(
        experiment_id=experiment_id,
        base_candidate_id=definition.base_candidate_id,
        baseline_candidate_id=baseline_id,
        dimension_paths=tuple(dimension.path for dimension in search_space.dimensions),
        candidate_ordinals=candidate_ordinals,
        runs=analysis_runs,
    )


def _select_runs(
    runs: Sequence[AnalysisRun],
    filters: Sequence[AnalysisFilter],
    qualities: Sequence[str],
) -> list[AnalysisRun]:
    allowed_qualities = set(qualities)
    return [
        run
        for run in runs
        if (not allowed_qualities or run.quality in allowed_qualities)
        and _matches_filters(run, filters)
    ]


def _matches_filters(
    run: AnalysisRun,
    filters: Sequence[AnalysisFilter],
) -> bool:
    return all(_resolve_path(run, item.path) == item.value for item in filters)


def _resolve_path(run: AnalysisRun, path: str | None) -> JsonScalar:
    if path is None:
        return None
    if path == "run.id":
        return run.run_id
    if path == "run.candidate_id":
        return run.candidate_id
    if path == "run.workload_case_id":
        return run.workload_case_id
    if path == "run.host_id":
        return run.host_id
    if path == "run.binary_id":
        return run.binary_id
    if path == "run.quality":
        return run.quality
    if path in {"workload.suite_case_index", "suite_case_index"}:
        return run.suite_case_index

    if path.startswith("workload."):
        payload: Any = run.workload.model_dump(mode="python", by_alias=True)
        segments = path.removeprefix("workload.").split(".")
    else:
        candidate_path = path.removeprefix("candidate.")
        payload = run.candidate.model_dump(mode="python", by_alias=True)
        segments = candidate_path.split(".")

    current: Any = payload
    for segment in segments:
        if not isinstance(current, Mapping) or segment not in current:
            raise AnalysisError(f"unknown analysis path: {path}")
        current = current[segment]
    if current is not None and not isinstance(current, (str, int, float, bool)):
        raise AnalysisError(f"analysis path is not scalar: {path}")
    return current


def _group_by_suite_case(
    runs: Sequence[AnalysisRun],
) -> dict[int, list[AnalysisRun]]:
    groups: dict[int, list[AnalysisRun]] = defaultdict(list)
    for run in runs:
        groups[run.suite_case_index].append(run)
    return groups


def _workload_label(run: AnalysisRun) -> str:
    workload = run.workload
    kind = getattr(workload, "kind", "workload")
    prompt = getattr(workload, "prompt_tokens", None)
    generated = getattr(workload, "generate_tokens", None)
    depth = getattr(workload, "depth_tokens", None)
    parts = [kind]
    if prompt is not None:
        parts.append(f"pp={prompt}")
    if generated is not None:
        parts.append(f"tg={generated}")
    if depth is not None:
        parts.append(f"depth={depth}")
    return " ".join(parts)


def _dominates(
    left: ParetoCandidate,
    right: ParetoCandidate,
    objectives: Sequence[ParetoObjective],
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


def _interpolate(
    points: Sequence[tuple[int, float]],
    tokens: int,
) -> float:
    if not points:
        raise AnalysisError("cannot interpolate an empty curve")
    if tokens <= points[0][0]:
        return points[0][1]
    if tokens >= points[-1][0]:
        return points[-1][1]
    for left, right in zip(points, points[1:], strict=False):
        if left[0] <= tokens <= right[0]:
            if right[0] == left[0]:
                return left[1]
            fraction = (tokens - left[0]) / (right[0] - left[0])
            return left[1] + (right[1] - left[1]) * fraction
    raise AssertionError("interpolation interval not found")


def _sorted_scalars(values: Iterable[JsonScalar]) -> tuple[JsonScalar, ...]:
    return tuple(sorted(values, key=_scalar_sort_key))


def _scalar_sort_key(value: JsonScalar) -> tuple[int, float | str]:
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, 1.0 if value else 0.0)
    if isinstance(value, (int, float)):
        return (2, float(value))
    return (3, value)


def _default_export_metrics() -> tuple[str, ...]:
    return (
        "throughput.mean",
        "throughput.median",
        "throughput.stddev",
        "throughput.cv",
        "throughput.min",
        "throughput.max",
        "throughput.sample_count",
        "cpu.process.avg_pct",
        "cpu.process.peak_pct",
        "cpu.system.avg_pct",
        "cpu.system.peak_pct",
        "memory.ram_used.peak_bytes",
        "memory.process_rss.peak_bytes",
        "gpu.utilization.avg_pct",
        "gpu.vram.peak_bytes",
        "gpu.temperature.peak_c",
        "gpu.power.avg_w",
    )


def _export_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value
