from __future__ import annotations

import shutil
import zipfile

from viam_sequence_to_lerobot.export_reader import load_export

from conftest import CAMERA, GOOD_SEQ, GOOD_TICKS, SHORT_SEQ, SHORT_TICKS


def test_load_export_from_directory(synthetic_export):
    export = load_export(synthetic_export)
    assert [s.sequence_id for s in export.sequences] == [GOOD_SEQ, SHORT_SEQ]
    assert export.sequences[0].duration_s == 2.0
    assert export.sequences[0].tags == ("session:test-good",)

    # The missing-file binary row is dropped; real files all resolve.
    camera_rows = export.binary_rows(GOOD_SEQ, CAMERA)
    assert len(camera_rows) == GOOD_TICKS
    assert all(r.path.is_file() for r in camera_rows)

    joints = export.tabular_rows(GOOD_SEQ, "xarm", "JointPositions")
    assert len(joints) == GOOD_TICKS - 1  # one tick intentionally has no reading
    assert len(joints[0].payload["positions"]["values"]) == 6

    # Rows are sorted by timestamp within each sequence.
    ts = [r.timestamp for r in camera_rows]
    assert ts == sorted(ts)


def test_load_export_from_zip(synthetic_export, tmp_path):
    zipped = tmp_path / "zipped-export"
    zipped.mkdir()
    shutil.copytree(synthetic_export / "binary_data", zipped / "binary_data")
    with zipfile.ZipFile(zipped / "archive.zip", "w") as zf:
        for name in ["sequences.parquet", "tabular_data.parquet", "binary_data.parquet"]:
            zf.write(synthetic_export / name, name)

    export = load_export(zipped)
    assert len(export.sequences) == 2
    assert len(export.binary_rows(SHORT_SEQ, CAMERA)) == SHORT_TICKS
