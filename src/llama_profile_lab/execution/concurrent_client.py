"""Synchronized llama-server concurrent workload client."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from threading import Event
from typing import Protocol

from llama_profile_lab.domain import (
    ConcurrentMemberResult,
    ConcurrentTokenEvent,
    ConcurrentWorkloadMemberSpec,
)


class ConcurrentClientError(RuntimeError):
    """Raised when a concurrent workload client cannot prepare or execute."""

    def __init__(self, message: str, *, kind: str = "failed") -> None:
        super().__init__(message)
        self.kind = kind


class PreparedConcurrentClient(Protocol):
    """Prepared request state that can start immediately after a barrier."""

    client_ready_ns: int

    def run(
        self,
        *,
        barrier_release_ns: int,
        cancel_event: Event | None = None,
    ) -> ConcurrentMemberResult:
        """Run the measured request."""


class ConcurrentClient(Protocol):
    """Prepare one workload client before synchronized release."""

    def prepare(
        self,
        endpoint: str,
        workload: ConcurrentWorkloadMemberSpec,
        *,
        cancel_event: Event | None = None,
    ) -> PreparedConcurrentClient:
        """Prepare any non-measured state and return a ready client."""


class LlamaCompletionConcurrentClient:
    """Use llama-server /completion streaming for measured concurrent work."""

    def prepare(
        self,
        endpoint: str,
        workload: ConcurrentWorkloadMemberSpec,
        *,
        cancel_event: Event | None = None,
    ) -> PreparedConcurrentClient:
        normalized = endpoint.rstrip("/")
        if workload.depth_tokens > 0:
            if cancel_event is not None and cancel_event.is_set():
                raise ConcurrentClientError(
                    "concurrent workload preparation cancelled",
                    kind="cancelled",
                )
            _request_json(
                f"{normalized}/completion",
                {
                    "prompt": [workload.prompt_token_id]
                    * workload.depth_tokens,
                    "n_predict": 0,
                    "cache_prompt": True,
                    "stream": False,
                    "ignore_eos": True,
                    "seed": 0,
                },
                timeout_seconds=workload.timeout_seconds,
            )
        return _PreparedLlamaCompletionClient(
            endpoint=normalized,
            workload=workload,
            client_ready_ns=time.monotonic_ns(),
        )


@dataclass(slots=True)
class _PreparedLlamaCompletionClient:
    endpoint: str
    workload: ConcurrentWorkloadMemberSpec
    client_ready_ns: int

    def run(
        self,
        *,
        barrier_release_ns: int,
        cancel_event: Event | None = None,
    ) -> ConcurrentMemberResult:
        workload = self.workload
        prompt_length = workload.depth_tokens + workload.prompt_tokens
        if prompt_length == 0:
            prompt_length = 1

        first_request_ns = time.monotonic_ns()
        prompt_events: list[ConcurrentTokenEvent] = []
        decode_events: list[ConcurrentTokenEvent] = []
        first_token_ns: int | None = None
        last_token_ns: int | None = None
        final_payload: dict[str, object] = {}
        status = "completed"
        failure: str | None = None

        payload = {
            "prompt": [workload.prompt_token_id] * prompt_length,
            "n_predict": workload.generate_tokens,
            "cache_prompt": True,
            "stream": True,
            "return_progress": True,
            "return_tokens": True,
            "timings_per_token": True,
            "ignore_eos": True,
            "seed": 0,
        }
        request = urllib.request.Request(
            f"{self.endpoint}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=workload.timeout_seconds,
            ) as response:
                for raw_line in response:
                    if cancel_event is not None and cancel_event.is_set():
                        raise ConcurrentClientError(
                            "concurrent workload cancelled",
                            kind="cancelled",
                        )
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError as exc:
                        raise ConcurrentClientError(
                            "llama-server emitted invalid SSE JSON",
                            kind="invalid",
                        ) from exc
                    if not isinstance(event, dict):
                        raise ConcurrentClientError(
                            "llama-server emitted a non-object SSE event",
                            kind="invalid",
                        )
                    now_ns = time.monotonic_ns()
                    final_payload = event
                    _append_prompt_progress(
                        prompt_events,
                        event,
                        now_ns=now_ns,
                        expected_tokens=workload.prompt_tokens,
                    )
                    predicted = _non_negative_int(
                        event.get("tokens_predicted")
                    )
                    if predicted is not None and predicted > 0:
                        if first_token_ns is None:
                            first_token_ns = now_ns
                        last_token_ns = now_ns
                        _append_cumulative(
                            decode_events,
                            kind="decode",
                            timestamp_ns=now_ns,
                            cumulative_tokens=predicted,
                        )
        except ConcurrentClientError as exc:
            status = (
                "cancelled"
                if exc.kind == "cancelled"
                else "invalid"
                if exc.kind == "invalid"
                else "failed"
            )
            failure = str(exc)
        except (TimeoutError, socket.timeout) as exc:
            status = "timeout"
            failure = str(exc) or "concurrent request timed out"
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                status = "timeout"
            else:
                status = "failed"
            failure = str(exc)
        except OSError as exc:
            status = "failed"
            failure = str(exc)

        finished_ns = time.monotonic_ns()
        timings = final_payload.get("timings")
        timing_map = timings if isinstance(timings, dict) else {}
        prompt_n_value = _non_negative_int(timing_map.get("prompt_n"))
        predicted_n_value = _non_negative_int(
            timing_map.get("predicted_n")
        )
        prompt_n = prompt_n_value or 0
        predicted_n = predicted_n_value or 0
        final_predicted = _non_negative_int(
            final_payload.get("tokens_predicted")
        )
        if (
            status == "completed"
            and final_predicted is not None
            and predicted_n_value is not None
            and final_predicted != predicted_n_value
        ):
            status = "invalid"
            failure = (
                "llama-server final token counters disagree: "
                f"stream={final_predicted}, timings={predicted_n_value}"
            )
        prompt_tps = _non_negative_float(
            timing_map.get("prompt_per_second")
        )
        decode_tps = _non_negative_float(
            timing_map.get("predicted_per_second")
        )

        concurrent_prompt_n = min(prompt_n, workload.prompt_tokens)
        if workload.mode == "prefill" and concurrent_prompt_n > 0:
            _append_cumulative(
                prompt_events,
                kind="prefill",
                timestamp_ns=finished_ns,
                cumulative_tokens=concurrent_prompt_n,
            )
        if status == "completed" and workload.mode == "decode" and predicted_n > 0:
            _append_cumulative(
                decode_events,
                kind="decode",
                timestamp_ns=last_token_ns or finished_ns,
                cumulative_tokens=predicted_n,
            )
            if first_token_ns is None:
                first_token_ns = last_token_ns or finished_ns
            if last_token_ns is None:
                last_token_ns = finished_ns

        correctness_valid = status == "completed"
        if status == "completed" and workload.require_exact_token_count:
            if workload.mode == "prefill":
                correctness_valid = concurrent_prompt_n == workload.prompt_tokens
            else:
                correctness_valid = predicted_n == workload.generate_tokens
            if not correctness_valid:
                status = "invalid"
                failure = "server token counts do not match requested workload"

        latency_ms = max(
            0.0,
            (finished_ns - first_request_ns) / 1_000_000.0,
        )
        all_events = tuple((*prompt_events, *decode_events))
        raw_summary = {
            "final_response_json": json.dumps(
                final_payload,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "timings_json": json.dumps(
                timing_map,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "token_events_json": json.dumps(
                [
                    event.model_dump(mode="json")
                    for event in all_events
                ],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "stop": bool(final_payload.get("stop", False)),
        }
        return ConcurrentMemberResult(
            instance_id=workload.instance_id,
            mode=workload.mode,
            status=status,
            client_ready_ns=self.client_ready_ns,
            barrier_release_ns=barrier_release_ns,
            first_request_ns=first_request_ns,
            first_token_ns=first_token_ns,
            last_token_ns=last_token_ns,
            finished_ns=finished_ns,
            prompt_tokens=concurrent_prompt_n,
            decode_tokens=predicted_n,
            native_prompt_tps=prompt_tps,
            native_decode_tps=decode_tps,
            latency_ms=latency_ms,
            token_events=all_events,
            correctness_valid=correctness_valid,
            failure=failure,
            raw=raw_summary,
        )


def _request_json(
    url: str,
    payload: dict[str, object],
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError) as exc:
        raise ConcurrentClientError(
            f"llama-server preparation failed: {exc}"
        ) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConcurrentClientError(
            "llama-server preparation returned invalid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ConcurrentClientError(
            "llama-server preparation response must be a JSON object"
        )
    return parsed


def _append_prompt_progress(
    events: list[ConcurrentTokenEvent],
    payload: dict[str, object],
    *,
    now_ns: int,
    expected_tokens: int,
) -> None:
    progress = payload.get("prompt_progress")
    if not isinstance(progress, dict):
        return
    processed = _non_negative_int(progress.get("processed"))
    cache = _non_negative_int(progress.get("cache")) or 0
    if processed is None:
        return
    measured = max(0, processed - cache)
    if expected_tokens >= 0:
        measured = min(measured, expected_tokens)
    _append_cumulative(
        events,
        kind="prefill",
        timestamp_ns=now_ns,
        cumulative_tokens=measured,
    )


def _append_cumulative(
    events: list[ConcurrentTokenEvent],
    *,
    kind: str,
    timestamp_ns: int,
    cumulative_tokens: int,
) -> None:
    if events:
        last = events[-1]
        if last.kind == kind:
            if cumulative_tokens < last.cumulative_tokens:
                raise ConcurrentClientError(
                    f"llama-server {kind} token counter regressed "
                    f"from {last.cumulative_tokens} to {cumulative_tokens}",
                    kind="invalid",
                )
            if last.cumulative_tokens == cumulative_tokens:
                return
    events.append(
        ConcurrentTokenEvent(
            kind=kind,
            timestamp_ns=timestamp_ns,
            cumulative_tokens=cumulative_tokens,
        )
    )


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _non_negative_float(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or value < 0
    ):
        return None
    return float(value)
