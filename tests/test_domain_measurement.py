"""Measurement-policy validation tests."""

import pytest
from pydantic import ValidationError

from llama_profile_lab.domain import AdaptiveMeasurementPolicy, MeasurementPolicy


def test_fixed_measurement_policy() -> None:
    policy = MeasurementPolicy(repetitions=7)
    assert policy.repetitions == 7
    assert policy.adaptive is None


def test_adaptive_policy_requires_no_fixed_repetition_count() -> None:
    adaptive = AdaptiveMeasurementPolicy(
        minimum_repetitions=3,
        maximum_repetitions=10,
        target_relative_error=0.01,
    )
    policy = MeasurementPolicy(repetitions=None, adaptive=adaptive)

    assert policy.adaptive == adaptive


def test_adaptive_and_fixed_cannot_be_combined() -> None:
    with pytest.raises(ValidationError, match="cannot also define fixed"):
        MeasurementPolicy(
            repetitions=3,
            adaptive=AdaptiveMeasurementPolicy(
                minimum_repetitions=3,
                maximum_repetitions=10,
                target_relative_error=0.01,
            ),
        )


def test_adaptive_bounds_are_validated() -> None:
    with pytest.raises(ValidationError, match="minimum_repetitions"):
        AdaptiveMeasurementPolicy(
            minimum_repetitions=10,
            maximum_repetitions=3,
            target_relative_error=0.01,
        )
