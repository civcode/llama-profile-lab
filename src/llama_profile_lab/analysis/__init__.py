"""Metrics, projections, Pareto analysis, and latency estimation."""

from llama_profile_lab.analysis.concurrent import (
    ConcurrentMetricError,
    MemberOverlap,
    compute_overlap,
    native_throughput,
    retention_for,
    tokens_completed_between,
)
from llama_profile_lab.analysis.metrics import (
    DEFAULT_METRIC_REGISTRY,
    MetricDefinition,
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
from llama_profile_lab.analysis.service import (
    AnalysisError,
    AnalysisService,
    serialize_export,
)

__all__ = [
    "ConcurrentMetricError",
    "MemberOverlap",
    "compute_overlap",
    "native_throughput",
    "retention_for",
    "tokens_completed_between",
    "AnalysisError",
    "AnalysisFilter",
    "AnalysisService",
    "BaselineDelta",
    "CandidateComparison",
    "CurvePoint",
    "DEFAULT_METRIC_REGISTRY",
    "LatencyEstimate",
    "MatrixCell",
    "MatrixFacet",
    "MatrixProjection",
    "MetricDefinition",
    "MetricRegistry",
    "ParetoCandidate",
    "ParetoObjective",
    "ParetoResult",
    "serialize_export",
]
