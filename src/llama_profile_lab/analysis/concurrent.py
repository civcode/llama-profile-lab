"""Overlap-aware concurrent deployment workload metrics."""

from __future__ import annotations

from dataclasses import dataclass

from llama_profile_lab.domain import (
    ConcurrentMemberResult,
    ConcurrentOverlapResult,
    ConcurrentRetention,
    ConcurrentTokenEvent,
)


class ConcurrentMetricError(ValueError):
    """Raised when synchronized results cannot produce valid overlap metrics."""


@dataclass(frozen=True, slots=True)
class MemberOverlap:
    """Per-member token counts normalized to the shared overlap interval."""

    instance_id: str
    prompt_tokens: int
    decode_tokens: int
    prompt_tps: float | None
    decode_tps: float | None


def compute_overlap(
    results: tuple[ConcurrentMemberResult, ...],
) -> tuple[ConcurrentOverlapResult, tuple[MemberOverlap, ...]]:
    """Compute exact event-counted throughput over the common active interval."""
    if len(results) < 2:
        raise ConcurrentMetricError(
            "concurrent overlap requires at least two member results"
        )
    if any(item.status != "completed" for item in results):
        raise ConcurrentMetricError(
            "concurrent overlap requires completed member results"
        )
    if any(
        item.first_request_ns is None or item.finished_ns is None
        for item in results
    ):
        raise ConcurrentMetricError(
            "concurrent overlap requires request and finish timestamps"
        )

    overlap_start = max(
        int(item.first_request_ns)
        for item in results
        if item.first_request_ns is not None
    )
    overlap_end = min(
        int(item.finished_ns)
        for item in results
        if item.finished_ns is not None
    )
    if overlap_end <= overlap_start:
        raise ConcurrentMetricError(
            "concurrent member intervals do not overlap"
        )
    duration_ns = overlap_end - overlap_start
    duration_seconds = duration_ns / 1_000_000_000.0

    members: list[MemberOverlap] = []
    prompt_total = 0
    decode_total = 0
    for result in results:
        prompt_tokens = tokens_completed_between(
            result.token_events,
            kind="prefill",
            start_ns=overlap_start,
            end_ns=overlap_end,
        )
        decode_tokens = tokens_completed_between(
            result.token_events,
            kind="decode",
            start_ns=overlap_start,
            end_ns=overlap_end,
        )
        prompt_total += prompt_tokens
        decode_total += decode_tokens
        members.append(
            MemberOverlap(
                instance_id=result.instance_id,
                prompt_tokens=prompt_tokens,
                decode_tokens=decode_tokens,
                prompt_tps=(
                    prompt_tokens / duration_seconds
                    if prompt_tokens > 0
                    else None
                ),
                decode_tps=(
                    decode_tokens / duration_seconds
                    if decode_tokens > 0
                    else None
                ),
            )
        )

    return (
        ConcurrentOverlapResult(
            overlap_start_ns=overlap_start,
            overlap_end_ns=overlap_end,
            overlap_duration_ns=duration_ns,
            prompt_tokens=prompt_total,
            decode_tokens=decode_total,
            combined_prompt_tps=(
                prompt_total / duration_seconds
                if prompt_total > 0
                else None
            ),
            combined_decode_tps=(
                decode_total / duration_seconds
                if decode_total > 0
                else None
            ),
        ),
        tuple(members),
    )


def tokens_completed_between(
    events: tuple[ConcurrentTokenEvent, ...],
    *,
    kind: str,
    start_ns: int,
    end_ns: int,
) -> int:
    """Return cumulative-token delta completed inside an inclusive interval."""
    before = 0
    at_end = 0
    for event in events:
        if event.kind != kind:
            continue
        if event.timestamp_ns < start_ns:
            before = max(before, event.cumulative_tokens)
        if event.timestamp_ns <= end_ns:
            at_end = max(at_end, event.cumulative_tokens)
    return max(0, at_end - before)


def native_throughput(result: ConcurrentMemberResult) -> float | None:
    """Select the member-native throughput matching its workload mode."""
    if result.mode == "prefill":
        return result.native_prompt_tps
    return result.native_decode_tps


def retention_for(
    result: ConcurrentMemberResult,
    *,
    baseline_id: str,
    standalone_tps: float,
) -> ConcurrentRetention:
    """Compute exact standalone/concurrent throughput retention."""
    concurrent_tps = native_throughput(result)
    if concurrent_tps is None:
        raise ConcurrentMetricError(
            f"member {result.instance_id} has no native {result.mode} throughput"
        )
    if standalone_tps <= 0:
        raise ConcurrentMetricError("standalone throughput must be positive")
    retention = concurrent_tps / standalone_tps
    return ConcurrentRetention(
        instance_id=result.instance_id,
        mode=result.mode,
        baseline_id=baseline_id,
        standalone_tps=standalone_tps,
        concurrent_tps=concurrent_tps,
        retention=retention,
        throughput_loss_pct=(1.0 - retention) * 100.0,
    )
