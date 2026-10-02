"""End-to-end managed llama-server + SPEED-Bench validation tests."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from llama_profile_lab.db import (
    CandidateRepository,
    Database,
    EnvironmentRepository,
    ExperimentRepository,
    MeasurementPolicyRepository,
    PlacementRepository,
    SearchSpaceRepository,
    ServerValidationRepository,
    WorkloadCaseRepository,
    WorkloadSuiteRepository,
)
from llama_profile_lab.domain import (
    Candidate,
    CandidateBaseline,
    ComputeConfig,
    ContextConfig,
    ExperimentDefinition,
    MeasurementPolicy,
    ModelSelection,
    PlacementConfig,
    ResolvedPlacement,
    SearchDimension,
    SearchSpace,
    ServerConfig,
    SpeculativeConfig,
    SpeedBenchConfig,
    SpeedBenchSuiteCase,
    SpeedBenchWorkloadCase,
    WorkloadSuite,
)
from llama_profile_lab.execution import ServerValidationService
from llama_profile_lab.execution.host import BasicHostInfo
from llama_profile_lab.llama import probe_binary


def write_fake_server(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

HELP = """usage: llama-server [options]
  --model F
  --host H
  --port N
  --ctx-size N
  --batch-size N
  --ubatch-size N
  --cache-type-k T
  --cache-type-v T
  --parallel N
  --n-gpu-layers N
  --kv-offload
  --no-kv-offload
  --kv-unified
  --no-kv-unified
  --op-offload
  --no-op-offload
  --repack
  --no-repack
  --alias NAME
  --spec-type TYPE
  --spec-draft-n-max N
  --spec-draft-model F
"""

if "--version" in sys.argv:
    print("version: 9001 (abcdef123)")
    raise SystemExit(0)
if "--help" in sys.argv or "-h" in sys.argv:
    print(HELP)
    raise SystemExit(0)

def option(name, default=None):
    if name not in sys.argv:
        return default
    return sys.argv[sys.argv.index(name) + 1]

host = option("--host", "127.0.0.1")
port = int(option("--port", "8080"))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

server = HTTPServer((host, port), Handler)
print(f"ready {host}:{port}", flush=True)
server.serve_forever()
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def write_failing_server(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
import sys

HELP = """usage: llama-server [options]
  --model F
  --host H
  --port N
  --ctx-size N
  --batch-size N
  --ubatch-size N
  --cache-type-k T
  --cache-type-v T
  --parallel N
  --n-gpu-layers N
  --kv-offload
  --no-kv-offload
  --kv-unified
  --no-kv-unified
  --op-offload
  --no-op-offload
  --repack
  --no-repack
"""

if "--version" in sys.argv:
    print("version: 9002 (deadbeef)")
    raise SystemExit(0)
if "--help" in sys.argv or "-h" in sys.argv:
    print(HELP)
    raise SystemExit(0)

print("synthetic server startup failure", file=sys.stderr)
raise SystemExit(42)
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def write_fake_speed_bench(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
import json
import sys

HELP = """usage: speed_bench.py [options]
  --url URL
  --model NAME
  --bench NAME
  --category NAME
  --osl N
  --extra-inputs JSON
  --concurrency N
  --limit N
  --timeout N
  --output FILE
"""

if "--version" in sys.argv:
    print("version: 1")
    raise SystemExit(0)
if "--help" in sys.argv or "-h" in sys.argv:
    print(HELP)
    raise SystemExit(0)

def option(name, default=None):
    if name not in sys.argv:
        return default
    return sys.argv[sys.argv.index(name) + 1]

model = option("--model", "baseline")
output = option("--output")
is_spec = model == "spec"
draft_n = 100 if is_spec else 0
accepted = 60 if is_spec else 0
payload = {
    "config": {
        "url": option("--url"),
        "model": model,
        "bench": option("--bench"),
        "category": option("--category"),
        "osl": int(option("--osl")),
        "concurrency": int(option("--concurrency")),
        "extra_inputs": json.loads(option("--extra-inputs", "{}")),
    },
    "selected_samples": 2,
    "completed_samples": 2,
    "failed_samples": 0,
    "summary": [{
        "category": "overall",
        "requests": 2,
        "turns": 2,
        "failed": 0,
        "avg_prompt_t_s": 105.0 if is_spec else 100.0,
        "avg_pred_t_s": 75.0 if is_spec else 50.0,
        "avg_latency": 1.2 if is_spec else 2.0,
        "draft_n": draft_n,
        "accepted": accepted,
        "accept_rate": (accepted / draft_n) if draft_n else None,
    }],
    "results": [],
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
print("speed_bench: wrote", output)
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def host_info() -> BasicHostInfo:
    return BasicHostInfo(
        hostname="server-validation-host",
        hardware_fingerprint="server-validation-hardware",
        cpu={"logical_cpus": 16},
        ram_bytes=64 * 1024**3,
        gpus=[{"pci_address": "0000:01:00.0"}],
        os_info={"system": "Linux"},
    )


def candidate(*, speculative: bool) -> Candidate:
    return Candidate(
        model=ModelSelection(
            target_model_id="model:target",
            draft_model_id="model:draft" if speculative else None,
        ),
        context=ContextConfig(
            size=8192,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        ),
        compute=ComputeConfig(batch_size=2048, ubatch_size=512),
        placement=PlacementConfig(mode="fixed"),
        server=ServerConfig(parallel=1),
        speculative=(
            SpeculativeConfig(enabled=True, type="draft-mtp", draft_n_max=3)
            if speculative
            else SpeculativeConfig()
        ),
    )


def register_binary(database: Database, path: Path) -> str:
    probe = probe_binary(path)
    with database.session() as connection:
        return EnvironmentRepository(connection).put_binary(
            sha256=probe.sha256,
            kind=probe.kind,
            path=str(probe.path),
            size_bytes=probe.size_bytes,
            mtime_ns=probe.mtime_ns,
            git_commit=probe.git_commit,
            build_number=probe.build_number,
            build_info=probe.build_info_mapping(),
            capabilities=probe.capabilities.to_mapping(),
        )


def seed_validation(
    database: Database,
) -> tuple[str, str, str, str, str]:
    baseline = candidate(speculative=False)
    spec = candidate(speculative=True)
    suite = WorkloadSuite(
        id="server-validation",
        cases=(
            SpeedBenchSuiteCase(
                speed_bench=SpeedBenchConfig(
                    bench="qualitative",
                    categories=("all",),
                    output_tokens=128,
                    concurrency=1,
                    limit=2,
                    request={"temperature": 0},
                )
            ),
        ),
    )
    concrete = SpeedBenchWorkloadCase(speed_bench=suite.cases[0].speed_bench)
    search = SearchSpace(
        dimensions=(SearchDimension(path="compute.batch_size", values=(2048,)),)
    )
    policy = MeasurementPolicy(repetitions=3)

    with database.session() as connection:
        candidates = CandidateRepository(connection)
        baseline_id = candidates.put(baseline)
        spec_id = candidates.put(spec)
        search_id = SearchSpaceRepository(connection).put(search)
        suite_id = WorkloadSuiteRepository(connection).put(suite)
        workload_id = WorkloadCaseRepository(connection).put(concrete)
        policy_id = MeasurementPolicyRepository(connection).put(policy)
        experiment_id = ExperimentRepository(connection).create(
            ExperimentDefinition(
                name="server validation",
                base_candidate_id=baseline_id,
                search_space_id=search_id,
                workload_suite_id=suite_id,
                measurement_policy_id=policy_id,
                baseline=CandidateBaseline(candidate_id=baseline_id),
            ),
            status="planned",
        )
        experiments = ExperimentRepository(connection)
        experiments.add_candidate(
            experiment_id=experiment_id,
            candidate_id=baseline_id,
            ordinal=0,
            generation_metadata={},
        )
        experiments.add_candidate(
            experiment_id=experiment_id,
            candidate_id=spec_id,
            ordinal=1,
            generation_metadata={},
        )
        for candidate_id in (baseline_id, spec_id):
            experiments.add_workload(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                workload_case_id=workload_id,
                suite_case_index=0,
                expansion_provenance={},
            )

        info = host_info()
        environment = EnvironmentRepository(connection)
        host_id = environment.put_host(
            hostname=info.hostname,
            hardware_fingerprint=info.hardware_fingerprint,
            cpu=info.cpu,
            ram_bytes=info.ram_bytes,
            gpus=info.gpus,
            os_info=info.os_info,
        )
        fit_binary_id = environment.put_binary(
            sha256="f" * 64,
            kind="llama-fit-params",
            path="/synthetic/llama-fit-params",
            size_bytes=1,
            mtime_ns=1,
        )
        placements = PlacementRepository(connection)
        placement_ids = []
        for index, candidate_id in enumerate((baseline_id, spec_id)):
            placement_ids.append(
                placements.put_resolved(
                    placement_hash=f"{index + 1:064x}",
                    candidate_id=candidate_id,
                    host_id=host_id,
                    binary_id=fit_binary_id,
                    fit_attempt_id=None,
                    placement=ResolvedPlacement(
                        production_context_size=8192,
                        n_gpu_layers=42,
                        n_cpu_moe=0,
                        split_mode="layer",
                        main_gpu=0,
                        devices="auto",
                        tensor_split=None,
                        override_tensor=(),
                    ),
                    request={},
                    argv=(),
                    stdout="",
                    stderr="",
                    exit_code=0,
                )
            )

    return experiment_id, baseline_id, spec_id, placement_ids[0], placement_ids[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_server_validation_persists_speculative_metrics_and_comparison(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "server.db")
    experiment_id, baseline_id, spec_id, baseline_place, spec_place = seed_validation(
        database
    )
    server = tmp_path / "llama-server"
    speed = tmp_path / "speed_bench.py"
    write_fake_server(server)
    write_fake_speed_bench(speed)
    server_id = register_binary(database, server)
    speed_id = register_binary(database, speed)

    model = tmp_path / "target.gguf"
    draft = tmp_path / "draft.gguf"
    model.write_bytes(b"target")
    draft.write_bytes(b"draft")

    service = ServerValidationService(database, host_detector=host_info)
    baseline = service.validate(
        experiment_id,
        candidate_id=baseline_id,
        server_binary_id=server_id,
        speed_bench_binary_id=speed_id,
        model_path=model,
        placement_id=baseline_place,
        model_name="baseline",
        port=free_port(),
        readiness_timeout_seconds=5,
        benchmark_timeout_seconds=5,
    )
    spec = service.validate(
        experiment_id,
        candidate_id=spec_id,
        server_binary_id=server_id,
        speed_bench_binary_id=speed_id,
        model_path=model,
        draft_model_path=draft,
        placement_id=spec_place,
        model_name="spec",
        port=free_port(),
        readiness_timeout_seconds=5,
        benchmark_timeout_seconds=5,
    )

    assert baseline.completed
    assert spec.completed
    comparison = service.compare(
        experiment_id,
        baseline_candidate_id=baseline_id,
        speculative_candidate_id=spec_id,
    )
    assert comparison.baseline_pred_ts == 50.0
    assert comparison.speculative_pred_ts == 75.0
    assert comparison.decode_speedup == pytest.approx(1.5)
    assert comparison.latency_speedup == pytest.approx(2.0 / 1.2)
    assert comparison.draft_n == 100
    assert comparison.accepted_n == 60
    assert comparison.accept_rate == pytest.approx(0.6)

    with database.session() as connection:
        repository = ServerValidationRepository(connection)
        spec_run = repository.get_run(spec.server_run_id)
        assert spec_run is not None
        assert spec_run.status == "completed"
        assert spec_run.spec_type == "draft-mtp"
        assert spec_run.spec_draft_n_max == 3

        benchmarks = repository.benchmarks_for_candidate(
            experiment_id=experiment_id,
            candidate_id=spec_id,
        )
        assert len(benchmarks) == 1
        assert benchmarks[0].avg_pred_ts == 75.0
        assert benchmarks[0].accept_rate == pytest.approx(0.6)

        evaluations = repository.evaluations(
            experiment_id=experiment_id,
            candidate_id=spec_id,
        )
        assert [evaluation.stage for evaluation in evaluations] == [
            "finalist",
            "server-validated",
        ]
        assert evaluations[-1].decision == "completed"



def test_server_start_failure_is_persisted_without_benchmark_rows(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "server-failure.db")
    experiment_id, baseline_id, _, placement_id, _ = seed_validation(database)
    server = tmp_path / "llama-server"
    speed = tmp_path / "speed_bench.py"
    write_failing_server(server)
    write_fake_speed_bench(speed)
    server_id = register_binary(database, server)
    speed_id = register_binary(database, speed)

    model = tmp_path / "target.gguf"
    model.write_bytes(b"target")

    summary = ServerValidationService(database, host_detector=host_info).validate(
        experiment_id,
        candidate_id=baseline_id,
        server_binary_id=server_id,
        speed_bench_binary_id=speed_id,
        model_path=model,
        placement_id=placement_id,
        port=free_port(),
        readiness_timeout_seconds=2,
        benchmark_timeout_seconds=2,
    )

    assert not summary.completed
    assert summary.benchmark_ids == ()

    with database.session() as connection:
        repository = ServerValidationRepository(connection)
        run = repository.get_run(summary.server_run_id)
        assert run is not None
        assert run.status == "start_failed"
        assert run.exit_code == 42
        assert repository.benchmarks_for_candidate(
            experiment_id=experiment_id,
            candidate_id=baseline_id,
            completed_only=False,
        ) == ()

        evaluations = repository.evaluations(
            experiment_id=experiment_id,
            candidate_id=baseline_id,
        )
        assert evaluations[-1].stage == "server-validated"
        assert evaluations[-1].decision == "failed"
        assert evaluations[-1].reason is not None
        assert "exited before readiness" in evaluations[-1].reason
