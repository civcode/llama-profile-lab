"""Matrix, baseline, Pareto, latency, and export analysis tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llama_profile_lab.analysis import (
    AnalysisError,
    AnalysisFilter,
    AnalysisService,
    ParetoObjective,
    serialize_export,
)
from llama_profile_lab.db import CandidateRepository
from tests.analysis_helpers import candidate_id_for, seed_analysis_experiment


def pp8k_filters() -> tuple[AnalysisFilter, ...]:
    return (
        AnalysisFilter(path="workload.kind", value="microbench-prefill"),
        AnalysisFilter(path="workload.prompt_tokens", value=8192),
    )


def tg4k_filters() -> tuple[AnalysisFilter, ...]:
    return (
        AnalysisFilter(path="workload.kind", value="microbench-decode"),
        AnalysisFilter(path="workload.depth_tokens", value=4096),
    )


def test_reference_matrix_is_sparse_and_supports_workload_facets(tmp_path: Path) -> None:
    database, experiment_id = seed_analysis_experiment(tmp_path / "matrix.db")
    service = AnalysisService(database)

    matrix = service.matrix(
        experiment_id,
        x_path="compute.batch_size",
        y_path="compute.ubatch_size",
        metric="throughput.median",
        filters=pp8k_filters(),
        qualities=("clean",),
    )

    assert matrix.x_values == (2048, 4096, 8192)
    assert matrix.y_values == (512, 1024, 2048, 4096)
    assert len(matrix.facets) == 1
    assert len(matrix.facets[0].cells) == 11
    assert not any(cell.x == 2048 and cell.y == 4096 for cell in matrix.facets[0].cells)

    faceted = service.matrix(
        experiment_id,
        x_path="compute.batch_size",
        y_path="compute.ubatch_size",
        metric="throughput.median",
        facet_path="workload.suite_case_index",
    )
    assert len(faceted.facets) == 4
    assert all(len(facet.cells) == 11 for facet in faceted.facets)


def test_reference_acceptance_matrices_cover_pp_tg_and_cpu(tmp_path: Path) -> None:
    database, experiment_id = seed_analysis_experiment(tmp_path / "acceptance.db")
    service = AnalysisService(database)
    views = (
        (
            "throughput.median",
            (
                AnalysisFilter(path="workload.kind", value="microbench-prefill"),
                AnalysisFilter(path="workload.prompt_tokens", value=2048),
            ),
        ),
        ("throughput.median", pp8k_filters()),
        ("throughput.median", tg4k_filters()),
        (
            "throughput.median",
            (
                AnalysisFilter(path="workload.kind", value="microbench-decode"),
                AnalysisFilter(path="workload.depth_tokens", value=65408),
            ),
        ),
        ("cpu.process.avg_pct", tg4k_filters()),
    )

    for metric, filters in views:
        projection = service.matrix(
            experiment_id,
            x_path="compute.batch_size",
            y_path="compute.ubatch_size",
            metric=metric,
            filters=filters,
        )
        assert len(projection.facets) == 1
        assert len(projection.facets[0].cells) == 11


def test_matrix_rejects_hidden_workload_dimensions(tmp_path: Path) -> None:
    database, experiment_id = seed_analysis_experiment(tmp_path / "ambiguous.db")

    with pytest.raises(AnalysisError, match="multiple Candidate/workload"):
        AnalysisService(database).matrix(
            experiment_id,
            x_path="compute.batch_size",
            y_path="compute.ubatch_size",
            metric="throughput.median",
        )


def test_baseline_comparison_reports_signed_deltas(tmp_path: Path) -> None:
    database, experiment_id = seed_analysis_experiment(tmp_path / "compare.db")
    candidate_id = candidate_id_for(
        database,
        experiment_id,
        batch=8192,
        ubatch=4096,
    )

    comparison = AnalysisService(database).compare(
        experiment_id,
        candidate_id=candidate_id,
        metric_names=("throughput.median", "cpu.process.avg_pct"),
        filters=pp8k_filters(),
    )

    assert len(comparison.deltas) == 2
    throughput = next(
        delta for delta in comparison.deltas if delta.metric == "throughput.median"
    )
    cpu = next(delta for delta in comparison.deltas if delta.metric == "cpu.process.avg_pct")
    assert throughput.percent_delta is not None and throughput.percent_delta > 0
    assert cpu.percent_delta is not None and cpu.percent_delta > 0


def test_pareto_frontier_keeps_only_lowest_ubatch_for_each_batch(tmp_path: Path) -> None:
    database, experiment_id = seed_analysis_experiment(tmp_path / "pareto.db")
    result = AnalysisService(database).pareto(
        experiment_id,
        objectives=(
            ParetoObjective(
                key="pp8k",
                direction="maximize",
                metric="throughput.median",
                filters=pp8k_filters(),
            ),
            ParetoObjective(
                key="tg4k",
                direction="maximize",
                metric="throughput.median",
                filters=tg4k_filters(),
            ),
            ParetoObjective(
                key="cpu",
                direction="minimize",
                metric="cpu.process.avg_pct",
                filters=tg4k_filters(),
            ),
            ParetoObjective(
                key="rss",
                direction="minimize",
                metric="memory.process_rss.peak_bytes",
                filters=tg4k_filters(),
            ),
        ),
    )

    assert result.evaluated_count == 11
    assert len(result.frontier) == 3
    with database.session() as connection:
        candidates = CandidateRepository(connection)
        coordinates = {
            (
                candidates.get(item.candidate_id).compute.batch_size,
                candidates.get(item.candidate_id).compute.ubatch_size,
            )
            for item in result.frontier
            if candidates.get(item.candidate_id) is not None
        }
    assert coordinates == {(2048, 512), (4096, 512), (8192, 512)}


def test_latency_model_interpolates_prefill_and_decode_curves(tmp_path: Path) -> None:
    database, experiment_id = seed_analysis_experiment(tmp_path / "latency.db")
    candidate_id = candidate_id_for(
        database,
        experiment_id,
        batch=4096,
        ubatch=512,
    )

    estimate = AnalysisService(database).latency(
        experiment_id,
        candidate_id=candidate_id,
        prompt_tokens=4096,
        generate_tokens=128,
        decode_start_depth_tokens=8192,
        qualities=("clean",),
    )

    assert len(estimate.prefill_curve) == 2
    assert len(estimate.decode_curve) == 2
    assert estimate.prefill_seconds > 0
    assert estimate.decode_seconds > 0
    assert estimate.total_seconds == pytest.approx(
        estimate.prefill_seconds + estimate.decode_seconds
    )


def test_json_and_csv_exports_include_dimensions_and_statistics(tmp_path: Path) -> None:
    database, experiment_id = seed_analysis_experiment(tmp_path / "export.db")
    rows = AnalysisService(database).export_rows(
        experiment_id,
        filters=pp8k_filters(),
        metric_names=("throughput.median", "cpu.process.avg_pct"),
    )

    assert len(rows) == 11
    assert "candidate.compute.batch_size" in rows[0]
    assert "candidate.compute.ubatch_size" in rows[0]
    assert rows[0]["throughput.median"] is not None

    json_payload = json.loads(serialize_export(rows, format_name="json"))
    csv_payload = serialize_export(rows, format_name="csv")
    assert len(json_payload) == 11
    assert "throughput.median" in csv_payload.splitlines()[0]
