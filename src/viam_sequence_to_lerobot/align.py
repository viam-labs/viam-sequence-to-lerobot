"""Time-align asynchronous capture streams onto a common frame clock.

The camera stream defines the frame clock (its capture timestamps are the
ticks). Every other stream is matched to each tick by nearest timestamp,
subject to a tolerance. Ticks that cannot be matched in every stream are
dropped.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from collections.abc import Sequence
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def nearest(sorted_timestamps: Sequence[float], target: float) -> int:
    """Index of the timestamp nearest to ``target`` in a sorted sequence."""
    if not sorted_timestamps:
        raise ValueError("nearest() requires a non-empty sequence")
    i = bisect_left(sorted_timestamps, target)
    if i == 0:
        return 0
    if i == len(sorted_timestamps):
        return len(sorted_timestamps) - 1
    before, after = sorted_timestamps[i - 1], sorted_timestamps[i]
    return i if (after - target) < (target - before) else i - 1


def match_stream(
    ticks: Sequence[float],
    stream: Sequence[tuple[float, T]],
    tolerance_s: float,
) -> list[T | None]:
    """Match each tick to the nearest stream item within ``tolerance_s``.

    ``stream`` must be sorted by timestamp. Returns one entry per tick,
    ``None`` where no item is close enough.
    """
    timestamps = [ts for ts, _ in stream]
    matched: list[T | None] = []
    for tick in ticks:
        if not timestamps:
            matched.append(None)
            continue
        i = nearest(timestamps, tick)
        if abs(timestamps[i] - tick) <= tolerance_s:
            matched.append(stream[i][1])
        else:
            matched.append(None)
    return matched


def align_streams(
    ticks: Sequence[float],
    streams: dict[str, Sequence[tuple[float, T]]],
    tolerance_s: float,
) -> tuple[list[int], dict[str, list[T]]]:
    """Align several streams onto the tick clock, keeping fully matched ticks.

    Returns the indices of ticks for which *every* stream had a match, and a
    dict of the matched items per stream (in kept-tick order). Drops are
    logged per stream at DEBUG and summarized by the caller.
    """
    per_stream = {name: match_stream(ticks, stream, tolerance_s) for name, stream in streams.items()}
    kept: list[int] = []
    for i in range(len(ticks)):
        misses = [name for name, matches in per_stream.items() if matches[i] is None]
        if misses:
            logger.debug("Dropping tick %d (t=%.3f): no match in %s", i, ticks[i], misses)
        else:
            kept.append(i)
    aligned = {name: [matches[i] for i in kept] for name, matches in per_stream.items()}
    return kept, aligned
