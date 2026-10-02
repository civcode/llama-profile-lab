"""Experiment-definition schema tests."""

from llama_profile_lab.domain import ExperimentDefinition, FixedPlacementPolicy


def test_experiment_defaults_to_per_candidate_placement() -> None:
    experiment = ExperimentDefinition(
        name="Flash Next B x UB",
        base_candidate_id="cand_base",
        search_space_id="space_1",
        workload_suite_id="suite_1",
        measurement_policy_id="measure_1",
    )

    assert experiment.placement_policy.type == "per-candidate"
    assert experiment.baseline.type == "base-candidate"


def test_fixed_placement_policy_is_supported() -> None:
    experiment = ExperimentDefinition(
        name="Controlled placement",
        base_candidate_id="cand_base",
        search_space_id="space_1",
        workload_suite_id="suite_1",
        measurement_policy_id="measure_1",
        placement_policy=FixedPlacementPolicy(placement_id="place_abc"),
    )

    assert experiment.placement_policy.type == "fixed"
    assert experiment.placement_policy.placement_id == "place_abc"
