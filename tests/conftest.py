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


LEFT_ARM, RIGHT_ARM = "left-arm", "right-arm"
LEFT_GRIPPER, RIGHT_GRIPPER = "left-gripper", "right-gripper"
BIMANUAL_SEQ = "33333333-3333-3333-3333-333333333333"
NO_GRIPPER_SEQ = "44444444-4444-4444-4444-444444444444"
WIDE_SEQ = "55555555-5555-5555-5555-555555555555"
BIMANUAL_TICKS = 12


def arm_values(tick: int, base: float, n: int = N_JOINTS) -> list[float]:
    return [base + tick + j * 0.01 for j in range(n)]


@pytest.fixture
def bimanual_export(tmp_path: Path) -> Path:
    """Two arms (JointPositions) and two grippers (DoCommand), three 12-tick sequences at 10 Hz.

    - BIMANUAL_SEQ: complete. Left arm values tick + j*0.01, right arm 100 + tick + j*0.01,
      left gripper 10*tick, right gripper 500 + tick; readings 1-4 ms after each tick.
    - NO_GRIPPER_SEQ: the right gripper was never captured (skipped).
    - WIDE_SEQ: the right arm reports 7 joints (a different layout, skipped by convert()).
    """
    export_dir = tmp_path / "bimanual"
    export_dir.mkdir()
    seqs = [BIMANUAL_SEQ, NO_GRIPPER_SEQ, WIDE_SEQ]
    starts = [BASE + timedelta(minutes=i) for i in range(len(seqs))]
    sequences = {
        "sequence_id": seqs,
        "tags": [["cmd:hand over the cube"]] * len(seqs),
        "start_at": starts,
        "end_at": [s + timedelta(seconds=2) for s in starts],
    }
    columns = ["sequence_id", "timestamp", "part_id", "component_name", "method_name"]
    tabular = {k: [] for k in [*columns, "payload"]}
    binary = {k: [] for k in [*columns, "path"]}

    def add(table, seq, ts, component, method, key, value):
        for k, v in zip([*columns, key], [seq, ts, "part-1", component, method, value]):
            table[k].append(v)

    for seq, start in zip(seqs, starts):
        for tick in range(BIMANUAL_TICKS):
            ts = start + timedelta(seconds=0.1 * tick)
            rel = f"binary_data/{seq}/cam/{tick:04d}.jpeg"
            _write_image(export_dir / rel, tick)
            add(binary, seq, ts, CAMERA, "GetImages", "path", rel)
            right_n = N_JOINTS + 1 if seq == WIDE_SEQ else N_JOINTS
            for i, (arm, base, n) in enumerate(((LEFT_ARM, 0.0, N_JOINTS), (RIGHT_ARM, 100.0, right_n))):
                payload = json.dumps({"positions": {"values": arm_values(tick, base, n)}})
                add(tabular, seq, ts + timedelta(milliseconds=1 + i), arm, "JointPositions", "payload", payload)
            for i, (gripper, value) in enumerate(((LEFT_GRIPPER, 10.0 * tick), (RIGHT_GRIPPER, 500.0 + tick))):
                if seq == NO_GRIPPER_SEQ and gripper == RIGHT_GRIPPER:
                    continue
                payload = json.dumps({"docommand_output": {"position": value, "is_moving": False}})
                add(tabular, seq, ts + timedelta(milliseconds=3 + i), gripper, "DoCommand", "payload", payload)

    pq.write_table(pa.table(sequences), export_dir / "sequences.parquet")
    pq.write_table(pa.table(tabular), export_dir / "tabular_data.parquet")
    pq.write_table(pa.table(binary), export_dir / "binary_data.parquet")
    return export_dir
