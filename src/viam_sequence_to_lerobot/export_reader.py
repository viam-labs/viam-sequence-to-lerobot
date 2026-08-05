"""Read a Viam sequence dataset export from disk.

An export directory contains:
- ``sequences.parquet``, ``tabular_data.parquet``, ``binary_data.parquet``
  (either at the top level or inside a single ``.zip`` archive), and
- a ``binary_data/`` directory with the captured binary files (e.g. jpegs),
  referenced by the relative ``path`` column of ``binary_data.parquet``.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

PARQUET_FILES = ("sequences.parquet", "tabular_data.parquet", "binary_data.parquet")


@dataclass(frozen=True)
class Sequence:
    """One recorded sequence (episode) as listed in sequences.parquet."""

    sequence_id: str
    tags: tuple[str, ...]
    start_at: float  # epoch seconds, UTC
    end_at: float

    @property
    def duration_s(self) -> float:
        return self.end_at - self.start_at


@dataclass(frozen=True)
class TabularRow:
    """One tabular capture (e.g. a joint-position reading)."""

    sequence_id: str
    timestamp: float  # epoch seconds, UTC
    component_name: str
    method_name: str
    payload: dict


@dataclass(frozen=True)
class BinaryRow:
    """One binary capture (e.g. a camera frame), with its on-disk location."""

    sequence_id: str
    timestamp: float  # epoch seconds, UTC
    component_name: str
    method_name: str
    path: Path  # absolute path to the binary file


@dataclass
class SequenceExport:
    """A fully loaded Viam sequence export."""

    export_dir: Path
    sequences: list[Sequence]
    tabular_by_sequence: dict[str, list[TabularRow]] = field(default_factory=dict)
    binary_by_sequence: dict[str, list[BinaryRow]] = field(default_factory=dict)

    def tabular_rows(
        self, sequence_id: str, component_name: str, method_name: str
    ) -> list[TabularRow]:
        return [
            r
            for r in self.tabular_by_sequence.get(sequence_id, [])
            if r.component_name == component_name and r.method_name == method_name
        ]

    def binary_rows(self, sequence_id: str, component_name: str) -> list[BinaryRow]:
        return [
            r
            for r in self.binary_by_sequence.get(sequence_id, [])
            if r.component_name == component_name
        ]


def _to_epoch(ts: datetime) -> float:
    return ts.timestamp()


def _read_parquet_tables(export_dir: Path) -> dict[str, list[dict]]:
    """Read the three export parquet files, from the directory or a zip inside it."""
    tables: dict[str, list[dict]] = {}
    if all((export_dir / name).exists() for name in PARQUET_FILES):
        logger.debug("Reading parquet files directly from %s", export_dir)
        for name in PARQUET_FILES:
            tables[name] = pq.read_table(export_dir / name).to_pylist()
        return tables

    zips = sorted(export_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"{export_dir} contains neither the export parquet files "
            f"{PARQUET_FILES} nor a .zip archive with them"
        )
    if len(zips) > 1:
        logger.warning("Multiple zip archives in %s, using %s", export_dir, zips[0].name)
    logger.debug("Reading parquet files from archive %s", zips[0])
    with zipfile.ZipFile(zips[0]) as zf:
        members = set(zf.namelist())
        missing = [name for name in PARQUET_FILES if name not in members]
        if missing:
            raise FileNotFoundError(f"{zips[0]} is missing export files: {missing}")
        for name in PARQUET_FILES:
            tables[name] = pq.read_table(io.BytesIO(zf.read(name))).to_pylist()
    return tables


def load_export(export_dir: str | Path) -> SequenceExport:
    """Load a Viam sequence export, resolving binary paths against the export dir.

    Binary rows whose file is missing on disk are dropped with a warning.
    """
    export_dir = Path(export_dir).expanduser().resolve()
    if not export_dir.is_dir():
        raise NotADirectoryError(f"Export directory not found: {export_dir}")

    tables = _read_parquet_tables(export_dir)

    sequences = [
        Sequence(
            sequence_id=row["sequence_id"],
            tags=tuple(row["tags"]),
            start_at=_to_epoch(row["start_at"]),
            end_at=_to_epoch(row["end_at"]),
        )
        for row in tables["sequences.parquet"]
    ]
    sequences.sort(key=lambda s: s.start_at)

    tabular_by_sequence: dict[str, list[TabularRow]] = defaultdict(list)
    n_bad_payloads = 0
    for row in tables["tabular_data.parquet"]:
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            n_bad_payloads += 1
            continue
        tabular_by_sequence[row["sequence_id"]].append(
            TabularRow(
                sequence_id=row["sequence_id"],
                timestamp=_to_epoch(row["timestamp"]),
                component_name=row["component_name"],
                method_name=row["method_name"],
                payload=payload,
            )
        )
    if n_bad_payloads:
        logger.warning("Dropped %d tabular rows with unparseable JSON payloads", n_bad_payloads)

    binary_by_sequence: dict[str, list[BinaryRow]] = defaultdict(list)
    n_missing_files = 0
    for row in tables["binary_data.parquet"]:
        path = export_dir / row["path"]
        if not path.is_file():
            n_missing_files += 1
            continue
        binary_by_sequence[row["sequence_id"]].append(
            BinaryRow(
                sequence_id=row["sequence_id"],
                timestamp=_to_epoch(row["timestamp"]),
                component_name=row["component_name"],
                method_name=row["method_name"],
                path=path,
            )
        )
    if n_missing_files:
        logger.warning(
            "Dropped %d binary rows whose files are missing under %s",
            n_missing_files,
            export_dir,
        )

    for rows in tabular_by_sequence.values():
        rows.sort(key=lambda r: r.timestamp)
    for rows in binary_by_sequence.values():
        rows.sort(key=lambda r: r.timestamp)

    export = SequenceExport(
        export_dir=export_dir,
        sequences=sequences,
        tabular_by_sequence=dict(tabular_by_sequence),
        binary_by_sequence=dict(binary_by_sequence),
    )
    logger.info(
        "Loaded export %s: %d sequences, %d tabular rows, %d binary rows",
        export_dir.name,
        len(sequences),
        sum(len(v) for v in tabular_by_sequence.values()),
        sum(len(v) for v in binary_by_sequence.values()),
    )
    return export
