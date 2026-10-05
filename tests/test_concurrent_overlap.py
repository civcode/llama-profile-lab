"""Overlap-aware concurrent throughput arithmetic tests."""

import pytest

from llama_profile_lab.analysis.concurrent import (
    ConcurrentMetricError,
    compute_overlap,
)
from llama_profile_lab.domain import ConcurrentMemberResult, ConcurrentTokenEvent


SECOND = 1_000_000_000


def _decode_result(
    instance_id: str,
    *,
    start: int,
    finish: int,
    events: tuple[tuple[int, int], ...],
) -> ConcurrentMemberResult:
    return ConcurrentMemberResult(
        instance_id=instance_id,
        mode="decode",
        status="completed",
        client_ready_ns=0,
        barrier_release_ns=0,
        first_request_ns=start,
        first_token_ns=events[0][0],
        last_token_ns=events[-1][0],
        finished_ns=finish,
        decode_tokens=events[-1][1],
        native_decode_tps=10.0,
        token_events=tuple(
            ConcurrentTokenEvent(
                kind="decode",
                timestamp_ns=timestamp,
                cumulative_tokens=count,
            )
            for timestamp, count in events
        ),
    )


def test_overlap_counts_only_tokens_completed_inside_common_interval() -> None:
    left = _decode_result(
        "a",
        start=0,
        finish=4 * SECOND,
        events=(
            (SECOND // 2, 1),
            (3 * SECOND // 2, 2),
            (5 * SECOND // 2, 3),
            (7 * SECOND // 2, 4),
        ),
    )
    right = _decode_result(
        "b",
        start=SECOND,
        finish=3 * SECOND,
        events=(
            (6 * SECOND // 5, 1),
            (11 * SECOND // 5, 2),
            (14 * SECOND // 5, 3),
        ),
    )

    overlap, members = compute_overlap((left, right))

    assert overlap.overlap_start_ns == SECOND
    assert overlap.overlap_end_ns == 3 * SECOND
    assert overlap.decode_tokens == 5
    assert overlap.combined_decode_tps == pytest.approx(2.5)
    assert [item.decode_tokens for item in members] == [2, 3]


def test_no_overlap_is_rejected() -> None:
    left = _decode_result(
        "a",
        start=0,
        finish=SECOND,
        events=((SECOND // 2, 1),),
    )
    right = _decode_result(
        "b",
        start=SECOND,
        finish=2 * SECOND,
        events=((3 * SECOND // 2, 1),),
    )

    with pytest.raises(ConcurrentMetricError, match="do not overlap"):
        compute_overlap((left, right))
