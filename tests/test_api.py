"""HTTP-only acceptance coverage for the local FastAPI milestone."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI
from starlette.types import Message, Receive, Scope, Send

from llama_profile_lab.api import create_app
from llama_profile_lab.db import (
    Database,
    EnvironmentRepository,
    PlacementRepository,
)
from llama_profile_lab.domain import (
    AbsoluteDepth,
    Candidate,
    ComputeConfig,
    ContextConfig,
    FitConfig,
    MeasurementPolicy,
    ModelSelection,
    PlacementConfig,
    PrefillSuiteCase,
    ResolvedPlacement,
    SearchDimension,
    SearchSpace,
    WorkloadSuite,
)
from llama_profile_lab.execution import detect_basic_host
from tests.test_execution_engine import (
    write_fake_fit_params,
    write_fake_llama_bench,
)
from tests.test_server_validation import (
    seed_validation,
    write_fake_server,
    write_fake_speed_bench,
)


@dataclass(frozen=True, slots=True)
class ApiResponse:
    status_code: int
    headers: dict[str, str]
    content: bytes

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")


def api_request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    query: list[tuple[str, str]] | None = None,
    timeout: float = 10.0,
) -> ApiResponse:
    return asyncio.run(
        _api_request(
            app,
            method,
            path,
            body=body,
            query=query,
            timeout=timeout,
        )
    )


async def _api_request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None,
    query: list[tuple[str, str]] | None,
    timeout: float,
) -> ApiResponse:
    raw_body = b"" if body is None else json.dumps(body).encode("utf-8")
    headers: list[tuple[bytes, bytes]] = []
    if body is not None:
        headers.append((b"content-type", b"application/json"))

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": urlencode(query or (), doseq=True).encode("ascii"),
        "root_path": "",
        "headers": headers,
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
    }
    received = False
    messages: list[Message] = []

    async def receive() -> Message:
        nonlocal received
        if not received:
            received = True
            return {
                "type": "http.request",
                "body": raw_body,
                "more_body": False,
            }
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        messages.append(message)

    await asyncio.wait_for(
        app(scope, Receive(receive), Send(send)),
        timeout=timeout,
    )
    starts = [message for message in messages if message["type"] == "http.response.start"]
    assert len(starts) == 1
    start = starts[0]
    response_headers = {
        key.decode("latin-1"): value.decode("latin-1")
        for key, value in start["headers"]
    }
    content = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return ApiResponse(
        status_code=int(start["status"]),
        headers=response_headers,
        content=content,
    )


def experiment_payload(name: str) -> dict[str, Any]:
    candidate = Candidate(
        model=ModelSelection(target_model_id="model:api-test"),
        context=ContextConfig(
            size=8192,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        ),
        compute=ComputeConfig(
            batch_size=2048,
            ubatch_size=512,
        ),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=256, min_context=4096),
        ),
    )
    search = SearchSpace(
        dimensions=(
            SearchDimension(
                path="compute.batch_size",
                values=(2048,),
            ),
        )
    )
    suite = WorkloadSuite(
        id=f"{name}-suite",
        cases=(
            PrefillSuiteCase(
                label="pp-128",
                prompt_tokens=128,
                depth=AbsoluteDepth(tokens=0),
            ),
        ),
    )
    policy = MeasurementPolicy(repetitions=1)
    return {
        "name": name,
        "base_candidate": candidate.model_dump(mode="json", by_alias=True),
        "search_space": search.model_dump(mode="json", by_alias=True),
        "workload_suite": suite.model_dump(mode="json", by_alias=True),
        "measurement_policy": policy.model_dump(mode="json", by_alias=True),
    }


def wait_for_experiment_status(
    app: FastAPI,
    experiment_id: str,
    statuses: set[str],
    *,
    timeout: float = 8.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = api_request(
            app,
            "GET",
            f"/api/experiments/{experiment_id}/progress",
        )
        assert response.status_code == 200
        last = response.json()
        if last["experiment_status"] in statuses:
            return last
        time.sleep(0.05)
    raise AssertionError(f"experiment did not reach {statuses}; last={last}")


def test_health_profiles_binary_registration_and_experiment_planning(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "api.db"
    launcher = tmp_path / "workstation.json"
    launcher.write_text(
        json.dumps(
            {
                "binaries": {"custom": "~/bin/llama-server"},
                "defaults": {"args": {"--host": "127.0.0.1", "--port": 8080}},
                "profiles": {
                    "flash": {"args": {"--flash-attn": "on"}},
                },
                "models": {
                    "demo": {
                        "binary": "custom",
                        "profiles": ["flash"],
                        "model": "~/models/demo.gguf",
                        "server_alias": "demo",
                        "args": {"--ctx-size": 8192},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    app = create_app(database_path, launcher_config_path=launcher)

    health = api_request(app, "GET", "/api/health")
    assert health.status_code == 200
    assert health.json()["schema_version"] == 5

    profiles = api_request(app, "GET", "/api/profiles")
    assert profiles.status_code == 200
    profile = profiles.json()["items"][0]
    assert profile["id"] == "demo"
    assert profile["args"]["--host"] == "127.0.0.1"
    assert profile["args"]["--flash-attn"] == "on"
    assert profile["args"]["--ctx-size"] == 8192

    bench = tmp_path / "llama-bench"
    write_fake_llama_bench(bench)
    inspected = api_request(
        app,
        "POST",
        "/api/binaries/inspect",
        body={"paths": [str(bench)]},
    )
    assert inspected.status_code == 201
    assert inspected.json()["items"][0]["kind"] == "llama-bench"
    assert len(api_request(app, "GET", "/api/binaries").json()["items"]) == 1
    assert api_request(app, "GET", "/api/models").json()["items"] == []

    created = api_request(
        app,
        "POST",
        "/api/experiments",
        body=experiment_payload("HTTP plan"),
    )
    assert created.status_code == 201
    experiment_id = created.json()["id"]
    assert created.json()["status"] == "draft"

    planned = api_request(
        app,
        "POST",
        f"/api/experiments/{experiment_id}/plan",
    )
    assert planned.status_code == 200
    assert planned.json()["candidate_count"] == 1
    assert planned.json()["benchmark_case_count"] == 1

    candidates = api_request(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/candidates",
    )
    assert candidates.status_code == 200
    assert len(candidates.json()["items"]) == 1

    clone = api_request(
        app,
        "POST",
        f"/api/experiments/{experiment_id}/clone",
        body={"name": "HTTP clone"},
    )
    assert clone.status_code == 201
    assert clone.json()["status"] == "draft"
    assert clone.json()["name"] == "HTTP clone"
    assert clone.json()["id"] != experiment_id


def test_http_execution_results_telemetry_and_sse(tmp_path: Path) -> None:
    database_path = tmp_path / "execution.db"
    app = create_app(database_path)

    created = api_request(
        app,
        "POST",
        "/api/experiments",
        body=experiment_payload("HTTP execution"),
    )
    experiment_id = created.json()["id"]
    assert api_request(
        app,
        "POST",
        f"/api/experiments/{experiment_id}/plan",
    ).status_code == 200

    bench = tmp_path / "llama-bench"
    fit = tmp_path / "llama-fit-params"
    write_fake_llama_bench(bench)
    write_fake_fit_params(fit)
    inspected = api_request(
        app,
        "POST",
        "/api/binaries/inspect",
        body={"paths": [str(bench), str(fit)]},
    )
    assert inspected.status_code == 201
    by_kind = {item["kind"]: item["id"] for item in inspected.json()["items"]}

    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    execution = {
        "binary_id": by_kind["llama-bench"],
        "fit_binary_id": by_kind["llama-fit-params"],
        "model_path": str(model),
        "telemetry_interval_ms": 500,
    }
    started = api_request(
        app,
        "POST",
        f"/api/experiments/{experiment_id}/run",
        body=execution,
    )
    assert started.status_code == 202
    assert started.json()["operation"]["status"] in {
        "running",
        "completed",
    }

    progress = wait_for_experiment_status(app, experiment_id, {"completed"})
    assert progress["completed_cases"] == 1
    assert progress["incomplete_cases"] == 0

    runs = api_request(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/runs",
    )
    assert runs.status_code == 200
    run_id = runs.json()["items"][0]["id"]

    detail = api_request(app, "GET", f"/api/runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "completed"
    assert len(detail.json()["samples"]) == 1
    assert "avg_ts" in detail.json()["metrics"]

    telemetry = api_request(app, "GET", f"/api/runs/{run_id}/telemetry")
    assert telemetry.status_code == 200
    assert len(telemetry.json()["samples"]) >= 2

    results = api_request(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/results",
        query=[("metric", "throughput.median")],
    )
    assert results.status_code == 200
    assert len(results.json()["rows"]) == 1

    matrix = api_request(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/matrix",
        query=[
            ("x", "compute.batch_size"),
            ("y", "compute.ubatch_size"),
            ("metric", "throughput.median"),
            ("filter", "workload.kind=microbench-prefill"),
        ],
    )
    assert matrix.status_code == 200
    assert len(matrix.json()["facets"][0]["cells"]) == 1

    events = api_request(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/events",
    )
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert "event: progress" in events.text
    assert '"experiment_status":"completed"' in events.text


def test_http_pause_resume_and_cancel(tmp_path: Path) -> None:
    database_path = tmp_path / "control.db"
    app = create_app(database_path)

    slow_dir = tmp_path / "slow"
    slow_dir.mkdir()
    bench = slow_dir / "llama-bench"
    fit = slow_dir / "llama-fit-params"
    write_fake_llama_bench(bench)
    write_fake_fit_params(fit)
    bench_text = bench.read_text(encoding="utf-8")
    bench_text = bench_text.replace("import sys\n", "import sys\nimport time\n")
    bench_text = bench_text.replace(
        'repetitions = int(option("--repetitions", "3"))',
        'time.sleep(0.75)\nrepetitions = int(option("--repetitions", "3"))',
    )
    bench.write_text(bench_text, encoding="utf-8")

    inspected = api_request(
        app,
        "POST",
        "/api/binaries/inspect",
        body={"paths": [str(bench), str(fit)]},
    )
    by_kind = {item["kind"]: item["id"] for item in inspected.json()["items"]}
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    execution = {
        "binary_id": by_kind["llama-bench"],
        "fit_binary_id": by_kind["llama-fit-params"],
        "model_path": str(model),
        "telemetry_interval_ms": 500,
    }

    first = api_request(
        app,
        "POST",
        "/api/experiments",
        body=experiment_payload("pause-resume"),
    ).json()["id"]
    assert api_request(app, "POST", f"/api/experiments/{first}/plan").status_code == 200
    assert api_request(
        app,
        "POST",
        f"/api/experiments/{first}/run",
        body=execution,
    ).status_code == 202
    paused = api_request(app, "POST", f"/api/experiments/{first}/pause")
    assert paused.status_code == 202
    wait_for_experiment_status(app, first, {"paused"})

    resumed = api_request(
        app,
        "POST",
        f"/api/experiments/{first}/resume",
        body=execution,
    )
    assert resumed.status_code == 202
    wait_for_experiment_status(app, first, {"completed"})

    second = api_request(
        app,
        "POST",
        "/api/experiments",
        body=experiment_payload("cancel"),
    ).json()["id"]
    assert api_request(app, "POST", f"/api/experiments/{second}/plan").status_code == 200
    assert api_request(
        app,
        "POST",
        f"/api/experiments/{second}/run",
        body=execution,
    ).status_code == 202
    cancelled = api_request(app, "POST", f"/api/experiments/{second}/cancel")
    assert cancelled.status_code == 202
    wait_for_experiment_status(app, second, {"cancelled"})


def test_http_candidate_server_validation(tmp_path: Path) -> None:
    database_path = tmp_path / "server-api.db"
    database = Database(database_path)
    experiment_id, _, spec_id, _, _ = seed_validation(database)

    host = detect_basic_host()
    with database.session() as connection:
        environment = EnvironmentRepository(connection)
        host_id = environment.put_host(
            hostname=host.hostname,
            hardware_fingerprint=host.hardware_fingerprint,
            cpu=host.cpu,
            ram_bytes=host.ram_bytes,
            gpus=host.gpus,
            os_info=host.os_info,
        )
        fit_binary_id = str(
            connection.execute(
                "SELECT binary_id FROM resolved_placement LIMIT 1"
            ).fetchone()[0]
        )
        placement_id = PlacementRepository(connection).put_resolved(
            placement_hash="9" * 64,
            candidate_id=spec_id,
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

    app = create_app(database_path)
    server = tmp_path / "llama-server"
    speed = tmp_path / "speed_bench.py"
    write_fake_server(server)
    write_fake_speed_bench(speed)
    inspected = api_request(
        app,
        "POST",
        "/api/binaries/inspect",
        body={"paths": [str(server), str(speed)]},
    )
    by_kind = {item["kind"]: item["id"] for item in inspected.json()["items"]}

    target = tmp_path / "target.gguf"
    draft = tmp_path / "draft.gguf"
    target.write_bytes(b"target")
    draft.write_bytes(b"draft")

    validated = api_request(
        app,
        "POST",
        f"/api/candidates/{spec_id}/validate",
        body={
            "experiment_id": experiment_id,
            "server_binary_id": by_kind["llama-server"],
            "speed_bench_binary_id": by_kind["speed-bench"],
            "model_path": str(target),
            "draft_model_path": str(draft),
            "placement_id": placement_id,
            "model_name": "spec",
            "port": _free_port(),
            "readiness_timeout_seconds": 5,
            "benchmark_timeout_seconds": 5,
        },
        timeout=10,
    )
    assert validated.status_code == 200
    assert validated.json()["completed"] is True
    assert validated.json()["speculative"] is True

    candidates = api_request(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/candidates",
    )
    spec = next(item for item in candidates.json()["items"] if item["id"] == spec_id)
    assert spec["server_validation_count"] == 1


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
