"""Synthetic Viam sequence export fixture.

Builds a small export on disk with:
- one "good" sequence tagged `cmd:open the lid`: 16 camera ticks at 10 Hz, joint/pose readings ~1-2 ms
  off each tick, except tick 7 which has no joint reading (alignment drop);
  a second "wrist-cam" camera covers every tick except tick 3,
- one "short" sequence: 3 ticks and no wrist-cam (filtered by min_frames,
  and by the missing camera in multi-camera runs),
- one binary row pointing at a missing file (reader drop).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

GOOD_SEQ = "11111111-1111-1111-1111-111111111111"
SHORT_SEQ = "22222222-2222-2222-2222-222222222222"
CAMERA = "webcam-teleop"
WRIST_CAMERA = "wrist-cam"
ARM = "xarm"
N_JOINTS = 6
GOOD_TICKS = 16
SHORT_TICKS = 3
DROPPED_JOINT_TICK = 7
DROPPED_WRIST_TICK = 3
IMAGE_SIZE = (32, 24)  # (width, height)

BASE = datetime(2026, 7, 15, 15, 0, 0, tzinfo=timezone.utc)


def _joints_payload(tick: int) -> str:
    return json.dumps({"positions": {"values": [float(tick) + j * 0.01 for j in range(N_JOINTS)]}})


def _pose_payload(tick: int) -> str:
    pose = {k: float(tick) + i for i, k in enumerate(["x", "y", "z", "o_x", "o_y", "o_z", "theta"])}
    return json.dumps({"pose": pose})


def _write_image(path: Path, tick: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(tick)
    pixels = rng.integers(0, 255, size=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="JPEG")


@pytest.fixture
def synthetic_export(tmp_path: Path) -> Path:
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    sequences = {
        "sequence_id": [GOOD_SEQ, SHORT_SEQ],
        "tags": [["session:test-good", "cmd:open the lid"], ["session:test-short"]],
        "start_at": [BASE, BASE + timedelta(minutes=1)],
        "end_at": [BASE + timedelta(seconds=2), BASE + timedelta(minutes=1, seconds=1)],
    }

    tabular = {k: [] for k in ["sequence_id", "timestamp", "part_id", "component_name", "method_name", "payload"]}
    binary = {k: [] for k in ["sequence_id", "timestamp", "part_id", "component_name", "method_name", "path"]}

    def add_tabular(seq: str, ts: datetime, method: str, payload: str) -> None:
        tabular["sequence_id"].append(seq)
        tabular["timestamp"].append(ts)
        tabular["part_id"].append("part-1")
        tabular["component_name"].append(ARM)
        tabular["method_name"].append(method)
        tabular["payload"].append(payload)

    def add_binary(
        seq: str, ts: datetime, camera: str, rel_path: str, write: bool = True, tick: int = 0
    ) -> None:
        binary["sequence_id"].append(seq)
        binary["timestamp"].append(ts)
        binary["part_id"].append("part-1")
        binary["component_name"].append(camera)
        binary["method_name"].append("GetImages")
        binary["path"].append(rel_path)
        if write:
            _write_image(export_dir / rel_path, tick)

    for seq, start, n_ticks in [(GOOD_SEQ, BASE, GOOD_TICKS), (SHORT_SEQ, BASE + timedelta(minutes=1), SHORT_TICKS)]:
        for tick in range(n_ticks):
            ts = start + timedelta(seconds=0.1 * tick)
            add_binary(seq, ts, CAMERA, f"binary_data/{seq}/cam/{tick:04d}.jpeg", tick=tick)
            if seq == GOOD_SEQ and tick != DROPPED_WRIST_TICK:
                add_binary(
                    seq,
                    ts + timedelta(milliseconds=3),
                    WRIST_CAMERA,
                    f"binary_data/{seq}/wrist/{tick:04d}.jpeg",
                    tick=tick,
                )
            if not (seq == GOOD_SEQ and tick == DROPPED_JOINT_TICK):
                add_tabular(seq, ts + timedelta(milliseconds=1), "JointPositions", _joints_payload(tick))
            add_tabular(seq, ts + timedelta(milliseconds=2), "EndPosition", _pose_payload(tick))

    # A binary row whose file does not exist: the reader must drop it.
    add_binary(GOOD_SEQ, BASE + timedelta(seconds=10), CAMERA, f"binary_data/{GOOD_SEQ}/cam/missing.jpeg", write=False)

    pq.write_table(pa.table(sequences), export_dir / "sequences.parquet")
    pq.write_table(pa.table(tabular), export_dir / "tabular_data.parquet")
    pq.write_table(pa.table(binary), export_dir / "binary_data.parquet")
    return export_dir
