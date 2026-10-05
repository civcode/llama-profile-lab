"""Machine-checkable V2 workstation acceptance coverage."""

from __future__ import annotations

import json
from pathlib import Path

import llama_profile_lab.acceptance as acceptance_module
from llama_profile_lab.acceptance import evaluate_deployment_acceptance
from llama_profile_lab.cli.main import main
from llama_profile_lab.db import Database
from tests.test_deployment_analysis import _seed


def _complete_payload() -> dict[str, object]:
    deployment_id = "deploy-test"
    split_id = "split-placement"
    baseline_id = "baseline-placement"
    phases = ("dd", "pp", "pd", "dp")
    workload_runs = [
        {
            "id": f"work-{phase}",
            "deployment_run_id": "run-split",
            "phase": phase,
            "status": "completed",
            "correctness_valid": 1,
            "min_retention": 0.8,
        }
        for phase in phases
    ]
    return {
        "format": "llprof-deployment-export-v1",
        "schema_version": 15,
        "deployment_id": deployment_id,
        "immutable": {
            "deployment_candidates": [
                {
                    "id": deployment_id,
                    "resource_policy_json": json.dumps(
                        {
                            "allow_swap": False,
                            "allow_cpu_offload": False,
                        }
                    ),
                }
            ],
            "deployment_instances": [
                {
                    "deployment_candidate_id": deployment_id,
                    "instance_id": "qwen",
                    "candidate_id": "candidate-qwen",
                    "model_artifact_id": "artifact:qwen",
                    "binary_id": "server-bin",
                },
                {
                    "deployment_candidate_id": deployment_id,
                    "instance_id": "flash",
                    "candidate_id": "candidate-flash",
                    "model_artifact_id": "artifact:flash",
                    "binary_id": "server-bin",
                },
            ],
            "candidates": [],
            "workload_suites": [],
            "models": [],
            "model_files": [],
        },
        "planning": {
            "plans": [{"memory_rejected": 1}],
            "cases": [],
            "rejections": [
                {"reason": "device_memory_exceeded", "stage": "memory"}
            ],
            "placements": [],
            "instance_placements": [
                {
                    "deployment_placement_id": baseline_id,
                    "deployment_candidate_id": deployment_id,
                    "instance_id": "qwen",
                    "resolved_placement_id": "baseline-qwen",
                },
                {
                    "deployment_placement_id": baseline_id,
                    "deployment_candidate_id": deployment_id,
                    "instance_id": "flash",
                    "resolved_placement_id": "baseline-flash",
                },
                {
                    "deployment_placement_id": split_id,
                    "deployment_candidate_id": deployment_id,
                    "instance_id": "qwen",
                    "resolved_placement_id": "split-qwen",
                },
                {
                    "deployment_placement_id": split_id,
                    "deployment_candidate_id": deployment_id,
                    "instance_id": "flash",
                    "resolved_placement_id": "split-flash",
                },
            ],
            "placement_memory": [
                {
                    "deployment_placement_id": split_id,
                    "instance_id": instance_id,
                    "device_id": device_id,
                    "model_bytes": 100,
                    "context_bytes": 50,
                    "compute_bytes": 25,
                }
                for instance_id in ("qwen", "flash")
                for device_id in ("gpu0", "gpu1")
            ],
            "device_allocations": [
                {
                    "deployment_placement_id": split_id,
                    "device_id": device_id,
                    "projected_bytes": 1000,
                    "projected_free_bytes": 700,
                }
                for device_id in ("gpu0", "gpu1")
            ],
            "resolved_placements": [
                {
                    "id": "baseline-qwen",
                    "devices_json": '["CUDA0"]',
                    "n_cpu_moe": 0,
                },
                {
                    "id": "baseline-flash",
                    "devices_json": '["Vulkan0"]',
                    "n_cpu_moe": 0,
                },
                {
                    "id": "split-qwen",
                    "devices_json": '["CUDA0","Vulkan0"]',
                    "n_cpu_moe": 0,
                },
                {
                    "id": "split-flash",
                    "devices_json": '["CUDA0","Vulkan0"]',
                    "n_cpu_moe": 0,
                },
            ],
            "placement_attempts": [],
            "memory_estimate_attempts": [],
            "memory_estimates": [
                {
                    "helper_binary_id": "helper-bin",
                    "candidate_id": "candidate-qwen",
                },
                {
                    "helper_binary_id": "helper-bin",
                    "candidate_id": "candidate-flash",
                },
            ],
            "memory_estimate_devices": [],
        },
        "execution": {
            "deployment_runs": [
                {
                    "id": "run-baseline",
                    "deployment_candidate_id": deployment_id,
                    "deployment_placement_id": baseline_id,
                    "status": "completed",
                },
                {
                    "id": "run-split",
                    "deployment_candidate_id": deployment_id,
                    "deployment_placement_id": split_id,
                    "status": "completed",
                },
            ],
            "deployment_run_members": [],
            "gpu_samples": [
                {
                    "deployment_run_id": "run-split",
                    "timestamp_ns": 1,
                    "gpu_json": json.dumps(
                        [
                            {
                                "device": "CUDA0",
                                "stable_device_key": "gpu0",
                                "vram_total_bytes": 2000,
                                "vram_used_bytes": 1200,
                            },
                            {
                                "device": "Vulkan0",
                                "stable_device_key": "gpu1",
                                "vram_total_bytes": 2000,
                                "vram_used_bytes": 1100,
                            },
                        ]
                    ),
                }
            ],
            "concurrent_workload_cases": [],
            "workload_runs": workload_runs,
            "workload_members": [],
            "standalone_baselines": [],
            "operations": [],
        },
        "promotion": {
            "proposals": [
                {
                    "id": "promotion-test",
                    "deployment_placement_id": split_id,
                    "sources_json": json.dumps(
                        [
                            {"instance_id": "qwen"},
                            {"instance_id": "flash"},
                        ]
                    ),
                }
            ]
        },
        "environment": {
            "binaries": [
                {
                    "id": "server-bin",
                    "kind": "llama-server",
                    "sha256": "a" * 64,
                    "git_commit": "server-commit",
                    "build_number": 1,
                },
                {
                    "id": "helper-bin",
                    "kind": "llama-memory-estimator",
                    "sha256": "b" * 64,
                    "git_commit": "helper-commit",
                    "build_number": 1,
                },
            ],
            "hosts": [],
            "accelerator_devices": [
                {
                    "logical_device_name": "CUDA0",
                    "backend": "cuda",
                    "mapping_status": "mapped",
                    "physical_device_key": "gpu0",
                    "vendor": "NVIDIA",
                },
                {
                    "logical_device_name": "Vulkan0",
                    "backend": "vulkan",
                    "mapping_status": "mapped",
                    "physical_device_key": "gpu1",
                    "vendor": "AMD",
                },
            ],
        },
    }


def test_acceptance_report_passes_machine_checks_with_complete_evidence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = _complete_payload()
    monkeypatch.setattr(
        acceptance_module,
        "export_deployment",
        lambda database, deployment_id: payload,
    )

    report = evaluate_deployment_acceptance(
        Database(tmp_path / "unused.db"),
        "deploy-test",
        minimum_phase_repetitions=1,
    )

    assert report["machine_checks_passed"] is True
    assert report["release_ready"] is False
    assert report["phase_counts"] == {"dd": 1, "pp": 1, "pd": 1, "dp": 1}
    statuses = {item["key"]: item["status"] for item in report["checks"]}
    assert all(
        status == "pass"
        for status in statuses.values()
        if status != "pending"
    )
    assert statuses["pareto_finalist_selection"] == "pending"
    assert statuses["target_hardware_model_identity"] == "pending"
    assert statuses["automated_clean_checkout_gates"] == "pending"
    assert {item["device_id"] for item in report["memory_deltas"]} == {
        "gpu0",
        "gpu1",
    }


def test_acceptance_cli_writes_failed_report_for_incomplete_evidence(
    tmp_path: Path,
) -> None:
    database, subjects = _seed(tmp_path)
    deployment_id = subjects["a"][0]
    output = tmp_path / "acceptance.json"

    result = main(
        [
            "deployment",
            "acceptance-report",
            deployment_id,
            "--minimum-phase-repetitions",
            "1",
            "--format",
            "json",
            "--output",
            str(output),
            "--database",
            str(database.path),
        ]
    )

    assert result == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == "llprof-deployment-acceptance-v1"
    assert payload["machine_checks_passed"] is False
    assert payload["release_ready"] is False
    statuses = {item["key"]: item["status"] for item in payload["checks"]}
    assert statuses["canonical_phase_repetitions"] == "fail"
    assert statuses["automated_clean_checkout_gates"] == "pending"
