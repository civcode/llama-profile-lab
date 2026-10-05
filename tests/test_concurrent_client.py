"""Production llama-server concurrent client protocol tests."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from llama_profile_lab.domain import ConcurrentWorkloadMemberSpec
from llama_profile_lab.execution.concurrent_client import (
    LlamaCompletionConcurrentClient,
)


class _Handler(BaseHTTPRequestHandler):
    requests: list[dict[str, object]] = []
    predicted_n = 0
    stream_mode = "normal"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        assert isinstance(payload, dict)
        self.__class__.requests.append(payload)

        if payload.get("stream") is False:
            body = json.dumps({"timings": {"prompt_n": 4}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        prompt_progress = {
            "prompt_progress": {
                "processed": 6,
                "cache": 4,
                "total": 6,
                "time_ms": 1.0,
            }
        }
        predicted = self.__class__.predicted_n
        final = {
            "tokens_predicted": predicted,
            "stop": True,
            "timings": {
                "prompt_n": 6,
                "prompt_per_second": 12.0,
                "predicted_n": predicted,
                "predicted_per_second": 4.0,
            },
        }
        mode = self.__class__.stream_mode
        if mode == "invalid-json":
            body = b"data: not-json\n\n"
        elif mode == "counter-regression":
            first = {"tokens_predicted": 2}
            second = {"tokens_predicted": 1}
            body = (
                f"data: {json.dumps(first)}\n\n"
                f"data: {json.dumps(second)}\n\n"
            ).encode("utf-8")
        elif mode == "counter-mismatch":
            final["tokens_predicted"] = 2
            final["timings"]["predicted_n"] = 1
            body = (
                f"data: {json.dumps(prompt_progress)}\n\n"
                f"data: {json.dumps(final)}\n\n"
            ).encode("utf-8")
        else:
            body = (
                f"data: {json.dumps(prompt_progress)}\n\n"
                f"data: {json.dumps(final)}\n\n"
            ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def _server(
    predicted_n: int,
    *,
    stream_mode: str = "normal",
) -> tuple[ThreadingHTTPServer, Thread]:
    _Handler.requests = []
    _Handler.predicted_n = predicted_n
    _Handler.stream_mode = stream_mode
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    return server, thread


def test_completion_client_prewarms_depth_and_records_prompt_events() -> None:
    server, thread = _server(predicted_n=0)
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        workload = ConcurrentWorkloadMemberSpec(
            ordinal=0,
            instance_id="qwen",
            mode="prefill",
            prompt_tokens=2,
            depth_tokens=4,
        )

        prepared = LlamaCompletionConcurrentClient().prepare(
            endpoint,
            workload,
        )
        result = prepared.run(barrier_release_ns=prepared.client_ready_ns)

        assert result.status == "completed"
        assert result.correctness_valid
        assert result.prompt_tokens == 2
        assert result.native_prompt_tps == 12.0
        assert result.token_events[-1].cumulative_tokens == 2
        assert "token_events_json" in result.raw
        assert len(_Handler.requests) == 2
        assert _Handler.requests[0]["prompt"] == [1, 1, 1, 1]
        assert _Handler.requests[0]["n_predict"] == 0
        assert _Handler.requests[1]["prompt"] == [1, 1, 1, 1, 1, 1]
        assert _Handler.requests[1]["return_progress"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_completion_client_marks_invalid_sse_json_as_invalid() -> None:
    server, thread = _server(predicted_n=2, stream_mode="invalid-json")
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        workload = ConcurrentWorkloadMemberSpec(
            ordinal=0,
            instance_id="qwen",
            mode="decode",
            generate_tokens=2,
            depth_tokens=4,
        )

        prepared = LlamaCompletionConcurrentClient().prepare(
            endpoint,
            workload,
        )
        result = prepared.run(barrier_release_ns=prepared.client_ready_ns)

        assert result.status == "invalid"
        assert not result.correctness_valid
        assert "invalid SSE JSON" in (result.failure or "")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_completion_client_rejects_regressing_token_counter() -> None:
    server, thread = _server(
        predicted_n=2,
        stream_mode="counter-regression",
    )
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        workload = ConcurrentWorkloadMemberSpec(
            ordinal=0,
            instance_id="qwen",
            mode="decode",
            generate_tokens=2,
            depth_tokens=4,
        )

        prepared = LlamaCompletionConcurrentClient().prepare(
            endpoint,
            workload,
        )
        result = prepared.run(barrier_release_ns=prepared.client_ready_ns)

        assert result.status == "invalid"
        assert not result.correctness_valid
        assert "counter regressed" in (result.failure or "")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_completion_client_rejects_contradictory_final_counters() -> None:
    server, thread = _server(
        predicted_n=2,
        stream_mode="counter-mismatch",
    )
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        workload = ConcurrentWorkloadMemberSpec(
            ordinal=0,
            instance_id="qwen",
            mode="decode",
            generate_tokens=2,
            depth_tokens=4,
        )

        prepared = LlamaCompletionConcurrentClient().prepare(
            endpoint,
            workload,
        )
        result = prepared.run(barrier_release_ns=prepared.client_ready_ns)

        assert result.status == "invalid"
        assert not result.correctness_valid
        assert "counters disagree" in (result.failure or "")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_completion_client_marks_decode_count_mismatch_invalid() -> None:
    server, thread = _server(predicted_n=1)
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        workload = ConcurrentWorkloadMemberSpec(
            ordinal=0,
            instance_id="flash",
            mode="decode",
            generate_tokens=2,
            depth_tokens=4,
        )

        prepared = LlamaCompletionConcurrentClient().prepare(
            endpoint,
            workload,
        )
        result = prepared.run(barrier_release_ns=prepared.client_ready_ns)

        assert result.status == "invalid"
        assert not result.correctness_valid
        assert result.decode_tokens == 1
        assert "do not match" in (result.failure or "")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
