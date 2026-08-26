"""Convert a Viam sequence export into a LeRobot v3 dataset.

Mapping (Viam capture -> LeRobot feature):
- camera ``GetImages`` frames        -> ``observation.images.<camera>``, one
  feature per configured camera; the first camera defines the frame clock
- ``action_space="joints"`` (default):
  arm ``JointPositions`` (N joints)  -> ``observation.state``;
  ``JointPositions`` at the next tick -> ``action`` (next-state-as-action)
- ``action_space="delta-ee"``:
  arm ``EndPosition`` as ``[x, y, z, rx, ry, rz]`` -> ``observation.state``;
  body-frame delta to the next tick's pose -> ``action``

Joint readings and the frames of every non-clock camera are matched to each
clock tick by nearest timestamp within ``tolerance_s``; ticks that cannot be
fully matched are dropped. Sequences missing any configured camera entirely
are skipped. The last tick of each episode is consumed as the final action
target and is not written as a frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .align import align_streams, median_spacing
from .export_reader import Sequence, SequenceExport, load_export
from .pose import POSE_NAMES, pose_delta, pose_vector

logger = logging.getLogger(__name__)


class EpisodeSkip(Exception):
    """Raised while building an episode when the sequence cannot be converted."""


@dataclass(frozen=True)
class ConversionConfig:
    export_dir: Path
    output_root: Path
    repo_id: str
    task: str
    camera_components: tuple[str, ...] = ("webcam-teleop",)
    arm_component: str = "xarm"
    action_space: str = "joints"
    fps: int = 10
    tolerance_s: float = 0.05
    min_frames: int = 10

    def __post_init__(self) -> None:
        if not self.camera_components:
            raise ValueError("At least one camera component is required")
        if self.action_space not in ("joints", "delta-ee"):
            raise ValueError(
                f"action_space must be 'joints' or 'delta-ee', got {self.action_space!r}"
            )

    @property
    def clock_camera(self) -> str:
        return self.camera_components[0]


def camera_feature_key(component_name: str) -> str:
    safe = component_name.replace("-", "_").replace(".", "_")
    return f"observation.images.{safe}"


@dataclass
class EpisodeFrames:
    """Fully aligned per-tick data for one episode, ready to be written."""

    sequence: Sequence
    images: dict[str, list[Path]]  # camera component -> one path per frame
    states: np.ndarray  # (n_frames, n_joints) float32
    actions: np.ndarray  # (n_frames, n_joints) float32, next-tick joints

    @property
    def n_frames(self) -> int:
        return len(self.actions)


@dataclass
class ConversionSummary:
    episodes_written: int = 0
    frames_written: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (sequence_id, reason)

    def skip(self, sequence_id: str, reason: str) -> None:
        logger.warning("Skipping sequence %s: %s", sequence_id, reason)
        self.skipped.append((sequence_id, reason))


# Fraction by which a stream's measured rate may differ from its reference
# (the clock camera vs --fps, other streams vs the clock) before warning.
RATE_MISMATCH_FRACTION = 0.2


def _warn_rate_mismatches(
    sequence_id: str,
    ticks: list[float],
    streams: dict[str, list],
    config: ConversionConfig,
) -> None:
    """Warn when measured capture rates disagree with --fps or each other.

    A clock camera slower/faster than --fps mistimes the dataset timeline and
    the next-state action horizon; a joint or camera stream slower than the
    clock gets duplicated across frames by nearest-timestamp matching (a
    zero-order hold), which for joints turns many actions into "don't move".
    Streams faster than the clock are fine — subsampling to the frame clock
    is standard — so only the slow direction warns.
    """
    clock_dt = median_spacing(ticks)
    if clock_dt is None or clock_dt <= 0:
        return
    expected_dt = 1.0 / config.fps
    if abs(clock_dt - expected_dt) > RATE_MISMATCH_FRACTION * expected_dt:
        logger.warning(
            "Sequence %s: clock camera %r captured at ~%.1f Hz but --fps is %d; "
            "video timing and the next-state action horizon will be wrong "
            "unless --fps matches the capture rate",
            sequence_id,
            config.clock_camera,
            1.0 / clock_dt,
            config.fps,
        )
    for name, stream in streams.items():
        dt = median_spacing([ts for ts, _ in stream])
        if dt is None or dt <= 0:
            continue
        if dt - clock_dt > RATE_MISMATCH_FRACTION * clock_dt:
            logger.warning(
                "Sequence %s: stream %r captured at ~%.1f Hz, slower than the "
                "clock camera at ~%.1f Hz; nearest-timestamp matching will "
                "reuse stale readings across frames or drop ticks",
                sequence_id,
                name,
                1.0 / dt,
                1.0 / clock_dt,
            )


def joint_values(payload: dict) -> list[float]:
    """Extract joint values from an arm JointPositions payload."""
    return [float(v) for v in payload["positions"]["values"]]


def joint_names(n_joints: int) -> list[str]:
    return [f"joint_{i}" for i in range(n_joints)]


def build_episode(
    export: SequenceExport, sequence: Sequence, config: ConversionConfig
) -> EpisodeFrames:
    """Align one sequence's streams onto the clock camera's ticks.

    Raises:
        EpisodeSkip: If a required stream is missing or too little of the
            sequence can be aligned.
    """
    camera_rows = {
        name: export.binary_rows(sequence.sequence_id, name)
        for name in config.camera_components
    }
    for name, rows in camera_rows.items():
        if not rows:
            raise EpisodeSkip(f"no frames from camera {name!r}")
    delta_ee = config.action_space == "delta-ee"
    arm_method = "EndPosition" if delta_ee else "JointPositions"
    arm_key = "ee" if delta_ee else "joints"
    arm_rows = export.tabular_rows(sequence.sequence_id, config.arm_component, arm_method)
    if not arm_rows:
        raise EpisodeSkip(f"no {arm_method} readings from arm {config.arm_component!r}")

    clock_rows = camera_rows[config.clock_camera]
    ticks = [r.timestamp for r in clock_rows]
    streams: dict[str, list] = {
        arm_key: [(r.timestamp, r) for r in arm_rows],
    }
    for name in config.camera_components[1:]:
        streams[name] = [(r.timestamp, r) for r in camera_rows[name]]

    _warn_rate_mismatches(sequence.sequence_id, ticks, streams, config)
    kept, aligned = align_streams(ticks, streams, config.tolerance_s)
    n_dropped = len(ticks) - len(kept)
    if n_dropped:
        logger.info(
            "Sequence %s: dropped %d/%d ticks without a full match within %.0f ms",
            sequence.sequence_id,
            n_dropped,
            len(ticks),
            config.tolerance_s * 1000,
        )
    if len(kept) < 2:
        raise EpisodeSkip("fewer than 2 fully aligned frames")

    if delta_ee:
        poses = np.stack([pose_vector(r.payload) for r in aligned["ee"]])
        states = poses[:-1].astype(np.float32)
        actions = np.stack(
            [pose_delta(poses[i], poses[i + 1]) for i in range(len(poses) - 1)]
        ).astype(np.float32)
    else:
        joints = np.array([joint_values(r.payload) for r in aligned["joints"]], dtype=np.float32)
        states = joints[:-1]
        actions = joints[1:]  # command for tick t is the measured joints at tick t+1
    images = {config.clock_camera: [clock_rows[i].path for i in kept[:-1]]}
    for name in config.camera_components[1:]:
        images[name] = [r.path for r in aligned[name][:-1]]
    return EpisodeFrames(
        sequence=sequence, images=images, states=states, actions=actions
    )


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _build_features(
    state_names: list[str], image_shapes: dict[str, tuple[int, int, int]]
) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
    }
    for camera, shape in image_shapes.items():
        features[camera_feature_key(camera)] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    return features


def convert(config: ConversionConfig) -> ConversionSummary:
    """Run the full conversion and return a summary of what was written."""
    # Imported here so that reader/align stay usable without lerobot installed.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    export = load_export(config.export_dir)
    summary = ConversionSummary()

    episodes: list[EpisodeFrames] = []
    state_dim: int | None = None
    for sequence in export.sequences:
        try:
            episode = build_episode(export, sequence, config)
        except EpisodeSkip as skip:
            summary.skip(sequence.sequence_id, str(skip))
            continue
        if episode.n_frames < config.min_frames:
            summary.skip(
                sequence.sequence_id,
                f"only {episode.n_frames} frames (min {config.min_frames})",
            )
            continue
        if state_dim is None:
            state_dim = episode.actions.shape[1]
        elif episode.actions.shape[1] != state_dim:
            summary.skip(
                sequence.sequence_id,
                f"state dim {episode.actions.shape[1]} != {state_dim}",
            )
            continue
        episodes.append(episode)

    if not episodes:
        raise ValueError("No convertible episodes found in the export")
    assert state_dim is not None
    state_names = (
        POSE_NAMES if config.action_space == "delta-ee" else joint_names(state_dim)
    )

    image_shapes = {
        camera: _load_image(episodes[0].images[camera][0]).shape
        for camera in config.camera_components
    }
    logger.info(
        "Converting %d episodes (%d frames total): state/action dim %d, cameras %s",
        len(episodes),
        sum(e.n_frames for e in episodes),
        state_dim,
        {c: "x".join(map(str, s)) for c, s in image_shapes.items()},
    )

    dataset = LeRobotDataset.create(
        repo_id=config.repo_id,
        fps=config.fps,
        features=_build_features(state_names, image_shapes),
        root=config.output_root,
    )
    try:
        for idx, episode in enumerate(episodes):
            n_added = 0
            n_bad_images = 0
            for i in range(episode.n_frames):
                frame_images = {}
                for camera in config.camera_components:
                    image = _load_image(episode.images[camera][i])
                    if image.shape != image_shapes[camera]:
                        break
                    frame_images[camera_feature_key(camera)] = image
                else:
                    dataset.add_frame(
                        {
                            "observation.state": episode.states[i],
                            "action": episode.actions[i],
                            "task": config.task,
                            **frame_images,
                        }
                    )
                    n_added += 1
                    continue
                n_bad_images += 1
            if n_bad_images:
                logger.warning(
                    "Episode %d (%s): skipped %d frames with unexpected image shape",
                    idx,
                    episode.sequence.sequence_id,
                    n_bad_images,
                )
            if n_added == 0:
                summary.skip(episode.sequence.sequence_id, "all frames rejected")
                continue
            dataset.save_episode()
            summary.episodes_written += 1
            summary.frames_written += n_added
            logger.info(
                "Saved episode %d/%d (%s): %d frames, tags=%s",
                summary.episodes_written,
                len(episodes),
                episode.sequence.sequence_id,
                n_added,
                list(episode.sequence.tags),
            )
    finally:
        dataset.finalize()

    logger.info(
        "Done: %d episodes / %d frames written to %s (%d sequences skipped)",
        summary.episodes_written,
        summary.frames_written,
        config.output_root,
        len(summary.skipped),
    )
    return summary
