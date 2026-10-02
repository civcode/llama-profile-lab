"""SQLite persistence, migrations, and repositories."""

from llama_profile_lab.db.connection import connect_database, transaction
from llama_profile_lab.db.database import Database
from llama_profile_lab.db.migrations import (
    Migration,
    MigrationError,
    discover_migrations,
    migrate,
    schema_version,
)
from llama_profile_lab.db.repositories import (
    BenchmarkCaseRepository,
    BenchmarkRunRepository,
    CandidateRepository,
    EnvironmentRepository,
    ExperimentRepository,
    MeasurementPolicyRepository,
    PlacementRepository,
    SearchSpaceRepository,
    TelemetryRepository,
    WorkloadCaseRepository,
    WorkloadSuiteRepository,
)

__all__ = [
    "BenchmarkCaseRepository",
    "BenchmarkRunRepository",
    "CandidateRepository",
    "Database",
    "EnvironmentRepository",
    "ExperimentRepository",
    "MeasurementPolicyRepository",
    "PlacementRepository",
    "Migration",
    "MigrationError",
    "SearchSpaceRepository",
    "TelemetryRepository",
    "WorkloadCaseRepository",
    "WorkloadSuiteRepository",
    "connect_database",
    "discover_migrations",
    "migrate",
    "schema_version",
    "transaction",
]
