"""Workload-domain validation tests."""

import pytest
from pydantic import ValidationError

from llama_profile_lab.domain import (
    DecodeWorkloadCase,
    FractionalDepth,
    PrefillSuiteCase,
    PrefillWorkloadCase,
    WorkloadEnvelope,
    WorkloadSuite,
)


def test_fractional_depth_bounds() -> None:
    assert FractionalDepth(value=0.5).value == 0.5

    with pytest.raises(ValidationError):
        FractionalDepth(value=1.01)


def test_workload_suite_accepts_symbolic_depth() -> None:
    suite = WorkloadSuite(
        id="interactive-v1",
        cases=(
            PrefillSuiteCase(
                label="pp-2k",
                prompt_tokens=2048,
                depth=FractionalDepth(value=0.5),
            ),
        ),
    )

    assert suite.cases[0].depth.type == "fraction"


def test_workload_hash_is_independent_of_measurement_policy() -> None:
    workload = DecodeWorkloadCase(generate_tokens=256, depth_tokens=32768)

    first_hash = workload.content_hash()
    second_hash = DecodeWorkloadCase(
        generate_tokens=256,
        depth_tokens=32768,
    ).content_hash()

    assert first_hash == second_hash


def test_context_envelope_rejects_overflow() -> None:
    case = PrefillWorkloadCase(prompt_tokens=8192, depth_tokens=60000)

    with pytest.raises(ValidationError, match="exceeds Candidate context"):
        WorkloadEnvelope(case=case, context_size=65536)


def test_context_envelope_accepts_exact_boundary() -> None:
    case = PrefillWorkloadCase(prompt_tokens=8192, depth_tokens=57344)
    envelope = WorkloadEnvelope(case=case, context_size=65536)

    assert envelope.context_size == 65536
