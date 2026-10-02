"""Safe constraint-expression tests."""

import pytest

from llama_profile_lab.planning import ConstraintError, evaluate_constraint


def candidate_payload() -> dict[str, object]:
    return {
        "compute": {
            "batch_size": 4096,
            "ubatch_size": 2048,
        },
        "context": {
            "cache_type_k": "q8_0",
        },
        "speculative": {
            "enabled": False,
        },
    }


def test_comparisons_boolean_logic_and_membership() -> None:
    payload = candidate_payload()

    assert evaluate_constraint(
        "compute.ubatch_size <= compute.batch_size", payload
    )
    assert evaluate_constraint(
        "context.cache_type_k in ('f16', 'q8_0') and speculative.enabled == false",
        payload,
    )
    assert not evaluate_constraint(
        "compute.batch_size < 2048 or speculative.enabled == true",
        payload,
    )


def test_function_calls_are_rejected() -> None:
    with pytest.raises(ConstraintError, match="unsupported constraint syntax"):
        evaluate_constraint("__import__('os').system('true')", candidate_payload())


def test_unknown_paths_are_rejected() -> None:
    with pytest.raises(ConstraintError, match="unknown Candidate path"):
        evaluate_constraint("compute.not_real == 1", candidate_payload())
