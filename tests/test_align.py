from __future__ import annotations

import pytest

from viam_sequence_to_lerobot.align import align_streams, match_stream, nearest


def test_nearest_picks_closest():
    ts = [0.0, 1.0, 2.0]
    assert nearest(ts, -5.0) == 0
    assert nearest(ts, 0.4) == 0
    assert nearest(ts, 0.6) == 1
    assert nearest(ts, 10.0) == 2


def test_nearest_empty_raises():
    with pytest.raises(ValueError):
        nearest([], 0.0)


def test_match_stream_respects_tolerance():
    stream = [(0.0, "a"), (1.0, "b")]
    assert match_stream([0.01, 0.5, 1.02], stream, tolerance_s=0.05) == ["a", None, "b"]
    assert match_stream([0.5], [], tolerance_s=0.05) == [None]


def test_align_streams_keeps_fully_matched_ticks():
    ticks = [0.0, 0.1, 0.2]
    kept, aligned = align_streams(
        ticks,
        {
            "joints": [(0.001, "j0"), (0.101, "j1"), (0.201, "j2")],
            "pose": [(0.002, "p0"), (0.202, "p2")],  # nothing near tick 1
        },
        tolerance_s=0.05,
    )
    assert kept == [0, 2]
    assert aligned == {"joints": ["j0", "j2"], "pose": ["p0", "p2"]}
