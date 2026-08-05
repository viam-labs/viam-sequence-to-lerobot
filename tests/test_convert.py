from __future__ import annotations

import numpy as np
import pytest
import torch

from viam_sequence_to_lerobot.convert import (
    ConversionConfig,
    EpisodeSkip,
    build_episode,
    convert,
)
from viam_sequence_to_lerobot.export_reader import load_export

from conftest import (
    CAMERA,
    DROPPED_JOINT_TICK,
    GOOD_TICKS,
    IMAGE_SIZE,
    N_JOINTS,
    SHORT_SEQ,
    WRIST_CAMERA,
)


def make_config(synthetic_export, tmp_path, **overrides) -> ConversionConfig:
    defaults = dict(
        export_dir=synthetic_export,
        output_root=tmp_path / "lerobot_out",
        repo_id="test/synthetic",
        task="open the box",
        fps=10,
        min_frames=10,
    )
    defaults.update(overrides)
    return ConversionConfig(**defaults)


def test_build_episode_aligns_and_shifts_actions(synthetic_export, tmp_path):
    export = load_export(synthetic_export)
    config = make_config(synthetic_export, tmp_path)
    episode = build_episode(export, export.sequences[0], config)

    # 16 ticks, one dropped (no joint reading), last one consumed for the action.
    assert episode.n_frames == GOOD_TICKS - 2
    assert episode.states.shape == (GOOD_TICKS - 2, N_JOINTS)
    assert episode.actions.shape == (GOOD_TICKS - 2, N_JOINTS)
    assert len(episode.images[CAMERA]) == GOOD_TICKS - 2
    # The action at each frame is the next kept state.
    np.testing.assert_allclose(episode.actions[:-1], episode.states[1:])
    # Tick 7 was dropped, so no state carries its joint marker value 7.0.
    assert float(DROPPED_JOINT_TICK) not in episode.states[:, 0]


def test_build_episode_multi_camera(synthetic_export, tmp_path):
    export = load_export(synthetic_export)
    config = make_config(
        synthetic_export, tmp_path, camera_components=(CAMERA, WRIST_CAMERA)
    )
    episode = build_episode(export, export.sequences[0], config)

    # One more tick dropped: the wrist camera has no frame at tick 3.
    assert episode.n_frames == GOOD_TICKS - 3
    assert set(episode.images) == {CAMERA, WRIST_CAMERA}
    assert len(episode.images[CAMERA]) == episode.n_frames
    assert len(episode.images[WRIST_CAMERA]) == episode.n_frames
    # Matched clock/wrist frames belong to the same tick (same file stem).
    for clock_path, wrist_path in zip(episode.images[CAMERA], episode.images[WRIST_CAMERA]):
        assert clock_path.stem == wrist_path.stem


def test_build_episode_missing_camera_skips(synthetic_export, tmp_path):
    export = load_export(synthetic_export)
    config = make_config(
        synthetic_export, tmp_path, camera_components=(CAMERA, WRIST_CAMERA)
    )
    short = next(s for s in export.sequences if s.sequence_id == SHORT_SEQ)
    with pytest.raises(EpisodeSkip, match=WRIST_CAMERA):
        build_episode(export, short, config)


def test_convert_end_to_end(synthetic_export, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = make_config(
        synthetic_export, tmp_path, camera_components=(CAMERA, WRIST_CAMERA)
    )
    summary = convert(config)

    assert summary.episodes_written == 1
    assert summary.frames_written == GOOD_TICKS - 3
    assert [seq_id for seq_id, _ in summary.skipped] == [SHORT_SEQ]

    dataset = LeRobotDataset(repo_id=config.repo_id, root=config.output_root)
    assert dataset.meta.total_episodes == 1
    assert len(dataset) == GOOD_TICKS - 3
    assert dataset.fps == 10
    assert set(dataset.meta.features) >= {
        "observation.state",
        "action",
        "observation.images.webcam_teleop",
        "observation.images.wrist_cam",
    }

    frame = dataset[0]
    assert frame["observation.state"].shape == (N_JOINTS,)
    assert frame["action"].shape == (N_JOINTS,)
    for key in ("observation.images.webcam_teleop", "observation.images.wrist_cam"):
        assert frame[key].shape[-2:] == (IMAGE_SIZE[1], IMAGE_SIZE[0])
    assert frame["task"] == "open the box"

    # action[t] == state[t+1] survives the round trip.
    torch.testing.assert_close(frame["action"], dataset[1]["observation.state"])


def test_convert_fails_without_episodes(synthetic_export, tmp_path):
    config = make_config(synthetic_export, tmp_path, camera_components=("nope",))
    with pytest.raises(ValueError, match="No convertible episodes"):
        convert(config)


def test_config_requires_a_camera(synthetic_export, tmp_path):
    with pytest.raises(ValueError, match="camera"):
        make_config(synthetic_export, tmp_path, camera_components=())
