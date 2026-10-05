"""Machine-checkable V2 deployment acceptance evidence."""

from __future__ import annotations

import json
from typing import Any

from llama_profile_lab.archive import export_deployment
from llama_profile_lab.db import Database

_REQUIRED_PHASES = ("dd", "pp", "pd", "dp")


def evaluate_deployment_acceptance(
    database: Database,
    deployment_id: str,
    *,
    minimum_devices: int = 2,
    minimum_phase_repetitions: int = 3,
) -> dict[str, Any]:
    """Evaluate persisted V2 workstation evidence without running workloads."""
    if minimum_devices < 1:
        raise ValueError("minimum_devices must be at least 1")
    if minimum_phase_repetitions < 1:
        raise ValueError("minimum_phase_repetitions must be at least 1")

    payload = export_deployment(database, deployment_id)
    immutable = payload["immutable"]
    planning = payload["planning"]
    execution = payload["execution"]
    environment = payload["environment"]

    root_instances = sorted(
        str(row["instance_id"])
        for row in immutable["deployment_instances"]
        if row["deployment_candidate_id"] == deployment_id
    )
    mapped = [
        row
        for row in environment["accelerator_devices"]
        if row.get("mapping_status") == "mapped" and _device_id(row)
    ]
    physical_devices = sorted(
        {device for row in mapped if (device := _device_id(row)) is not None}
    )
    backends = sorted({str(row["backend"]) for row in mapped if row.get("backend")})
    vendors = sorted({str(row["vendor"]) for row in mapped if row.get("vendor")})

    checks: list[dict[str, Any]] = []

    def add(key: str, passed: bool, summary: str, evidence: dict[str, Any]) -> None:
        checks.append(
            {
                "key": key,
                "status": "pass" if passed else "fail",
                "summary": summary,
                "evidence": evidence,
            }
        )

    add(
        "stable_device_inventory",
        len(physical_devices) >= minimum_devices,
        f"{len(physical_devices)} mapped physical devices have stable identity",
        {"physical_devices": physical_devices, "minimum_devices": minimum_devices},
    )
    heterogeneous = len(vendors) >= 2 or len(backends) >= 2
    add(
        "heterogeneous_device_inventory",
        heterogeneous,
        "multiple accelerator vendors/backends are persisted",
        {"vendors": vendors, "backends": backends},
    )

    telemetry = _simultaneous_telemetry(
        execution["gpu_samples"], set(physical_devices), minimum_devices
    )
    add(
        "simultaneous_multi_device_telemetry",
        telemetry is not None,
        "one telemetry sample covers the required mapped devices",
        {} if telemetry is None else telemetry,
    )

    identity = _identity_evidence(payload, deployment_id)
    add(
        "exact_binary_model_helper_identity",
        bool(identity["complete"]),
        "model, server, and estimator identities are content-addressed",
        identity,
    )

    memory_placements = _memory_placements(
        planning["placement_memory"],
        planning["device_allocations"],
        root_instances,
    )
    add(
        "projected_per_device_memory",
        bool(memory_placements),
        "per-instance model/context/compute memory is persisted",
        {"placement_ids": memory_placements},
    )

    memory_rejected = sum(
        int(row.get("memory_rejected") or 0) for row in planning["plans"]
    )
    memory_pruned = memory_rejected > 0 or any(
        "memory" in str(row.get("reason") or "").lower()
        for row in planning["rejections"]
    )
    add(
        "known_overcommit_pruned",
        memory_pruned,
        "planning evidence contains a memory rejection",
        {
            "memory_rejected": memory_rejected,
            "rejection_count": len(planning["rejections"]),
        },
    )

    phase_counts = {
        phase: sum(
            1
            for row in execution["workload_runs"]
            if row.get("phase") == phase
            and row.get("status") == "completed"
            and bool(row.get("correctness_valid"))
        )
        for phase in _REQUIRED_PHASES
    }
    add(
        "canonical_phase_repetitions",
        all(count >= minimum_phase_repetitions for count in phase_counts.values()),
        "DD/PP/PD/DP have the required correctness-valid repetitions",
        {
            "counts": phase_counts,
            "minimum_phase_repetitions": minimum_phase_repetitions,
        },
    )

    retention_phases = sorted(
        str(row["phase"])
        for row in execution["workload_runs"]
        if row.get("phase") in _REQUIRED_PHASES
        and row.get("status") == "completed"
        and bool(row.get("correctness_valid"))
        and row.get("min_retention") is not None
    )
    add(
        "standalone_to_concurrent_retention",
        set(retention_phases) == set(_REQUIRED_PHASES),
        "retention evidence exists for every canonical phase",
        {"phases": sorted(set(retention_phases))},
    )

    topology = _topologies(
        planning["instance_placements"],
        planning["resolved_placements"],
        root_instances,
        minimum_devices,
    )
    completed = {
        str(row["deployment_placement_id"])
        for row in execution["deployment_runs"]
        if row.get("status") == "completed" and row.get("deployment_placement_id")
    }
    completed_baseline = sorted(set(topology["baseline"]) & completed)
    completed_split = sorted(set(topology["split"]) & completed)
    add(
        "one_model_per_device_baseline",
        bool(completed_baseline),
        "a completed one-model-per-device baseline is persisted",
        {
            "completed_placement_ids": completed_baseline,
            "planned_placement_ids": topology["baseline"],
        },
    )
    add(
        "both_models_split_topology",
        bool(completed_split),
        "a completed topology has every instance spanning multiple devices",
        {
            "completed_placement_ids": completed_split,
            "planned_placement_ids": topology["split"],
        },
    )

    policy = _root_policy(immutable["deployment_candidates"], deployment_id)
    no_fallback = (
        policy.get("allow_swap") is False
        and policy.get("allow_cpu_offload") is False
        and all(
            row.get("n_cpu_moe") in (None, 0)
            for row in planning["resolved_placements"]
        )
    )
    add(
        "no_unintended_swap_or_cpu_offload",
        no_fallback,
        "policy forbids swap/CPU offload and placements have no CPU MoE",
        {
            "allow_swap": policy.get("allow_swap"),
            "allow_cpu_offload": policy.get("allow_cpu_offload"),
        },
    )

    promotion = _promotion_evidence(payload["promotion"]["proposals"], root_instances)
    add(
        "coordinated_promotion",
        bool(promotion["complete"]),
        "one coordinated promotion proposal covers every instance",
        promotion,
    )

    memory_deltas = _runtime_memory_deltas(payload)
    add(
        "runtime_memory_validation",
        bool(memory_deltas),
        "runtime GPU memory can be compared with projected placement memory",
        {"devices": sorted({row["device_id"] for row in memory_deltas})},
    )

    checks.extend(
        [
            _pending(
                "pareto_finalist_selection",
                "record explicit Pareto objectives/constraints and selected finalists",
                {},
            ),
            _pending(
                "target_hardware_model_identity",
                "confirm the reported devices/models are the intended target artifacts",
                {"instances": root_instances, "physical_devices": physical_devices},
            ),
            _pending(
                "automated_clean_checkout_gates",
                "backend/frontend clean-checkout gates must be recorded separately",
                {
                    "backend": [
                        "uv sync --frozen",
                        "uv run --frozen ruff check .",
                        "uv run --frozen mypy src",
                        "uv run --frozen pytest",
                    ],
                    "frontend": [
                        "npm ci",
                        "npm run typecheck",
                        "npm run test",
                        "npm run build",
                    ],
                },
            ),
        ]
    )

    machine_checks_passed = not any(row["status"] == "fail" for row in checks)
    return {
        "format": "llprof-deployment-acceptance-v1",
        "deployment_id": deployment_id,
        "criteria": {
            "minimum_devices": minimum_devices,
            "minimum_phase_repetitions": minimum_phase_repetitions,
            "required_phases": list(_REQUIRED_PHASES),
        },
        "machine_checks_passed": machine_checks_passed,
        "release_ready": machine_checks_passed
        and all(row["status"] == "pass" for row in checks),
        "checks": checks,
        "inventory": {
            "instances": root_instances,
            "physical_devices": physical_devices,
            "vendors": vendors,
            "backends": backends,
            "binaries": [
                {
                    "id": row.get("id"),
                    "kind": row.get("kind"),
                    "sha256": row.get("sha256"),
                    "git_commit": row.get("git_commit"),
                    "build_number": row.get("build_number"),
                }
                for row in environment["binaries"]
            ],
        },
        "phase_counts": phase_counts,
        "memory_deltas": memory_deltas,
    }


def serialize_deployment_acceptance_report(
    database: Database,
    deployment_id: str,
    *,
    minimum_devices: int = 2,
    minimum_phase_repetitions: int = 3,
) -> str:
    report = evaluate_deployment_acceptance(
        database,
        deployment_id,
        minimum_devices=minimum_devices,
        minimum_phase_repetitions=minimum_phase_repetitions,
    )
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _pending(key: str, summary: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"key": key, "status": "pending", "summary": summary, "evidence": evidence}


def _device_id(row: dict[str, Any]) -> str | None:
    for key in ("physical_device_key", "pci_bus_id", "uuid"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _decode(value: Any, fallback: Any) -> Any:
    if not isinstance(value, str):
        return fallback if value is None else value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _simultaneous_telemetry(
    rows: list[dict[str, Any]],
    expected_devices: set[str],
    minimum_devices: int,
) -> dict[str, Any] | None:
    for row in rows:
        gpus = _decode(row.get("gpu_json"), [])
        if not isinstance(gpus, list):
            continue
        observed = {
            str(item.get("stable_device_key") or item.get("device"))
            for item in gpus
            if isinstance(item, dict)
            and (item.get("stable_device_key") or item.get("device"))
        }
        matched = observed & expected_devices if expected_devices else observed
        if len(matched) >= minimum_devices:
            return {
                "deployment_run_id": row.get("deployment_run_id"),
                "timestamp_ns": row.get("timestamp_ns"),
                "devices": sorted(observed),
                "matched_physical_devices": sorted(matched),
            }
    return None


def _identity_evidence(payload: dict[str, Any], deployment_id: str) -> dict[str, Any]:
    instances = [
        row
        for row in payload["immutable"]["deployment_instances"]
        if row["deployment_candidate_id"] == deployment_id
    ]
    estimates = payload["planning"]["memory_estimates"]
    binaries = {str(row["id"]): row for row in payload["environment"]["binaries"]}
    candidate_ids = {
        str(row["candidate_id"])
        for row in payload["planning"]["resolved_placements"]
        if row.get("candidate_id")
    }
    estimated_ids = {
        str(row["candidate_id"]) for row in estimates if row.get("candidate_id")
    }
    helper_ids = {
        str(row["helper_binary_id"]) for row in estimates if row.get("helper_binary_id")
    }
    server_ids = {str(row["binary_id"]) for row in instances}
    missing_hashes = sorted(
        binary_id
        for binary_id in server_ids | helper_ids
        if binary_id not in binaries or not binaries[binary_id].get("sha256")
    )
    missing_artifacts = sorted(
        str(row["instance_id"])
        for row in instances
        if not row.get("model_artifact_id")
    )
    return {
        "complete": candidate_ids <= estimated_ids
        and bool(helper_ids)
        and not missing_hashes
        and not missing_artifacts,
        "server_binary_ids": sorted(server_ids),
        "helper_binary_ids": sorted(helper_ids),
        "model_artifact_ids": sorted(
            str(row["model_artifact_id"])
            for row in instances
            if row.get("model_artifact_id")
        ),
        "missing_binary_hashes": missing_hashes,
        "missing_model_artifact_instances": missing_artifacts,
        "estimated_candidate_ids": sorted(estimated_ids),
        "required_candidate_ids": sorted(candidate_ids),
    }


def _memory_placements(
    rows: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
    root_instances: list[str],
) -> list[str]:
    allocated = {str(row["deployment_placement_id"]) for row in allocations}
    seen: dict[str, set[str]] = {}
    for row in rows:
        if any(
            row.get(key) is None
            for key in ("model_bytes", "context_bytes", "compute_bytes")
        ):
            continue
        seen.setdefault(str(row["deployment_placement_id"]), set()).add(
            str(row["instance_id"])
        )
    return sorted(
        placement_id
        for placement_id, instances in seen.items()
        if placement_id in allocated and set(root_instances) <= instances
    )


def _topologies(
    rows: list[dict[str, Any]],
    resolved_rows: list[dict[str, Any]],
    root_instances: list[str],
    minimum_devices: int,
) -> dict[str, list[str]]:
    resolved: dict[str, set[str]] = {}
    for row in resolved_rows:
        raw = _decode(row.get("devices_json"), [])
        devices = raw if isinstance(raw, list) else []
        resolved[str(row["id"])] = {str(item) for item in devices if item}
    grouped: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["deployment_placement_id"]), {})[
            str(row["instance_id"])
        ] = resolved.get(str(row["resolved_placement_id"]), set())

    baseline: list[str] = []
    split: list[str] = []
    for placement_id, instances in grouped.items():
        if not set(root_instances) <= set(instances):
            continue
        device_sets = [instances[instance_id] for instance_id in root_instances]
        if all(len(devices) == 1 for devices in device_sets):
            if len(set().union(*device_sets)) >= min(minimum_devices, len(root_instances)):
                baseline.append(placement_id)
        if all(len(devices) >= minimum_devices for devices in device_sets):
            split.append(placement_id)
    return {"baseline": sorted(baseline), "split": sorted(split)}


def _root_policy(rows: list[dict[str, Any]], deployment_id: str) -> dict[str, Any]:
    for row in rows:
        if row.get("id") == deployment_id:
            value = _decode(row.get("resource_policy_json"), {})
            return value if isinstance(value, dict) else {}
    return {}


def _promotion_evidence(
    rows: list[dict[str, Any]], root_instances: list[str]
) -> dict[str, Any]:
    for row in reversed(rows):
        sources = _decode(row.get("sources_json"), [])
        if not isinstance(sources, list):
            continue
        instances = sorted(
            str(item["instance_id"])
            for item in sources
            if isinstance(item, dict) and item.get("instance_id")
        )
        if instances == root_instances:
            return {
                "complete": True,
                "proposal_id": row.get("id"),
                "deployment_placement_id": row.get("deployment_placement_id"),
                "instances": instances,
            }
    return {"complete": False, "proposal_id": None, "instances": []}


def _runtime_memory_deltas(payload: dict[str, Any]) -> list[dict[str, Any]]:
    allocations = {
        (str(row["deployment_placement_id"]), str(row["device_id"])): row
        for row in payload["planning"]["device_allocations"]
    }
    runs = {
        str(row["id"]): row
        for row in payload["execution"]["deployment_runs"]
        if row.get("status") == "completed" and row.get("deployment_placement_id")
    }
    peaks: dict[tuple[str, str], int] = {}
    minimum_free: dict[tuple[str, str], int] = {}
    for row in payload["execution"]["gpu_samples"]:
        run_id = str(row["deployment_run_id"])
        if run_id not in runs:
            continue
        gpus = _decode(row.get("gpu_json"), [])
        if not isinstance(gpus, list):
            continue
        for gpu in gpus:
            if not isinstance(gpu, dict):
                continue
            device_id = gpu.get("stable_device_key")
            used = gpu.get("vram_used_bytes")
            total = gpu.get("vram_total_bytes")
            if not isinstance(device_id, str) or not isinstance(used, int):
                continue
            key = (run_id, device_id)
            peaks[key] = max(peaks.get(key, used), used)
            if isinstance(total, int):
                free = total - used
                minimum_free[key] = min(minimum_free.get(key, free), free)

    result: list[dict[str, Any]] = []
    for (run_id, device_id), peak in sorted(peaks.items()):
        placement_id = str(runs[run_id]["deployment_placement_id"])
        allocation = allocations.get((placement_id, device_id))
        if allocation is None:
            continue
        projected = int(allocation["projected_bytes"])
        projected_free = int(allocation["projected_free_bytes"])
        runtime_free = minimum_free.get((run_id, device_id))
        result.append(
            {
                "deployment_run_id": run_id,
                "deployment_placement_id": placement_id,
                "device_id": device_id,
                "projected_bytes": projected,
                "runtime_peak_used_bytes": peak,
                "used_delta_bytes": peak - projected,
                "projected_free_bytes": projected_free,
                "runtime_min_free_bytes": runtime_free,
                "free_delta_bytes": None
                if runtime_free is None
                else runtime_free - projected_free,
            }
        )
    return result
