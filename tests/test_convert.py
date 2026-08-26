from __future__ import annotations

import numpy as np
import pytest
import torch

from viam_sequence_to_lerobot.align import median_spacing
from viam_sequence_to_lerobot.convert import (
    ConversionConfig,
    EpisodeSkip,
    _warn_rate_mismatches,
    build_episode,
    convert,
)
from viam_sequence_to_lerobot.export_reader import load_export
from viam_sequence_to_lerobot.pose import ACTION_NAMES, STATE_NAMES, state_compose

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


def test_build_episode_delta_ee(synthetic_export, tmp_path):
    export = load_export(synthetic_export)
    config = make_config(synthetic_export, tmp_path, action_space="delta-ee")
    episode = build_episode(export, export.sequences[0], config)

    # EndPosition covers every tick, including the one with no joint reading,
    # so only the final tick is consumed (as the last delta target).
    assert episode.n_frames == GOOD_TICKS - 1
    assert episode.states.shape == (GOOD_TICKS - 1, 9)
    assert episode.actions.shape == (GOOD_TICKS - 1, 6)
    # Composing each state with its delta action reproduces the next state.
    for t in range(episode.n_frames - 1):
        np.testing.assert_allclose(
            state_compose(
                episode.states[t].astype(np.float64),
                episode.actions[t].astype(np.float64),
            ),
            episode.states[t + 1],
            atol=1e-4,
        )


def test_build_episode_delta_ee_state_is_continuous(synthetic_export, tmp_path):
    # Consecutive states must move no further than the tool actually moved.
    # Guards the schema against a discontinuous rotation encoding, which the
    # compose round trip above cannot see.
    export = load_export(synthetic_export)
    config = make_config(synthetic_export, tmp_path, action_space="delta-ee")
    episode = build_episode(export, export.sequences[0], config)

    state_jumps = np.linalg.norm(np.diff(episode.states[:, 3:], axis=0), axis=1)
    action_rotations = np.linalg.norm(episode.actions[:-1, 3:], axis=1)
    assert np.all(state_jumps <= 2 * action_rotations + 1e-5)


def test_build_episode_delta_ee_missing_endposition_skips(synthetic_export, tmp_path):
    export = load_export(synthetic_export)
    config = make_config(
        synthetic_export, tmp_path, action_space="delta-ee", arm_component="no-such-arm"
    )
    with pytest.raises(EpisodeSkip, match="EndPosition"):
        build_episode(export, export.sequences[0], config)


def test_build_episode_delta_ee_skips_unusable_endposition(synthetic_export, tmp_path):
    # A zero orientation vector must skip the sequence, not write NaN or abort
    # the whole run.
    export = load_export(synthetic_export)
    config = make_config(synthetic_export, tmp_path, action_space="delta-ee")
    rows = export.tabular_rows(
        export.sequences[0].sequence_id, config.arm_component, "EndPosition"
    )
    rows[0].payload["pose"].update(o_x=0.0, o_y=0.0, o_z=0.0)
    with pytest.raises(EpisodeSkip, match="unusable EndPosition"):
        build_episode(export, export.sequences[0], config)


def test_convert_end_to_end_delta_ee(synthetic_export, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = make_config(synthetic_export, tmp_path, action_space="delta-ee")
    summary = convert(config)

    assert summary.episodes_written == 1
    assert summary.frames_written == GOOD_TICKS - 1

    dataset = LeRobotDataset(repo_id=config.repo_id, root=config.output_root)
    assert dataset.meta.features["observation.state"]["names"] == STATE_NAMES
    assert dataset.meta.features["action"]["names"] == ACTION_NAMES

    first, second = dataset[0], dataset[1]
    assert first["observation.state"].shape == (9,)
    assert first["action"].shape == (6,)
    composed = state_compose(
        first["observation.state"].numpy().astype(np.float64),
        first["action"].numpy().astype(np.float64),
    )
    np.testing.assert_allclose(composed, second["observation.state"].numpy(), atol=1e-4)


def test_config_rejects_unknown_action_space(synthetic_export, tmp_path):
    with pytest.raises(ValueError, match="action_space"):
        make_config(synthetic_export, tmp_path, action_space="bogus")


def test_convert_fails_without_episodes(synthetic_export, tmp_path):
    config = make_config(synthetic_export, tmp_path, camera_components=("nope",))
    with pytest.raises(ValueError, match="No convertible episodes"):
        convert(config)


def test_config_requires_a_camera(synthetic_export, tmp_path):
    with pytest.raises(ValueError, match="camera"):
        make_config(synthetic_export, tmp_path, camera_components=())


def test_median_spacing():
    assert median_spacing([]) is None
    assert median_spacing([1.0]) is None
    assert median_spacing([0.0, 0.1, 0.2, 0.3]) == pytest.approx(0.1)
    # Robust to one outlier gap (a dropped frame).
    assert median_spacing([0.0, 0.1, 0.2, 0.7, 0.8]) == pytest.approx(0.1)


def test_rate_check_warns_on_fps_mismatch(synthetic_export, tmp_path, caplog):
    config = make_config(synthetic_export, tmp_path, fps=30)
    ticks = [i * 0.1 for i in range(20)]  # 10 Hz clock, --fps 30
    with caplog.at_level("WARNING"):
        _warn_rate_mismatches("seq", ticks, {}, config)
    assert "but --fps is 30" in caplog.text


def test_rate_check_warns_on_slow_stream(synthetic_export, tmp_path, caplog):
    config = make_config(synthetic_export, tmp_path)
    ticks = [i * 0.1 for i in range(20)]  # 10 Hz clock, matching --fps
    joints = [(i * 0.2, object()) for i in range(10)]  # 5 Hz joints
    with caplog.at_level("WARNING"):
        _warn_rate_mismatches("seq", ticks, {"joints": joints}, config)
    assert "'joints'" in caplog.text
    assert "stale readings" in caplog.text


def test_rate_check_silent_on_fast_stream(synthetic_export, tmp_path, caplog):
    config = make_config(synthetic_export, tmp_path)
    ticks = [i * 0.1 for i in range(20)]  # 10 Hz clock
    joints = [(i * 0.02, object()) for i in range(100)]  # 50 Hz joints: benign
    with caplog.at_level("WARNING"):
        _warn_rate_mismatches("seq", ticks, {"joints": joints}, config)
    assert caplog.text == ""


def test_rate_check_silent_when_rates_match(synthetic_export, tmp_path, caplog):
    export = load_export(synthetic_export)
    config = make_config(synthetic_export, tmp_path)
    with caplog.at_level("WARNING"):
        build_episode(export, export.sequences[0], config)
    assert "Hz" not in caplog.text
