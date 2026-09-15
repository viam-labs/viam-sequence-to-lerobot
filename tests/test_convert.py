from __future__ import annotations

import numpy as np
import pytest
import torch

from viam_sequence_to_lerobot.align import median_spacing
from viam_sequence_to_lerobot.convert import (
    ConversionConfig,
    EpisodeSkip,
    _downscaled_size,
    _load_image,
    _warn_rate_mismatches,
    build_episode,
    convert,
    sequence_task,
)
from viam_sequence_to_lerobot.export_reader import Sequence, load_export
from viam_sequence_to_lerobot.pose import ACTION_NAMES, STATE_NAMES, state_compose

from conftest import (
    CAMERA,
    GOOD_SEQ,
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
    # With no gaps every written frame is the next tick, so the compose chain
    # closes frame to frame. That is a property of gapless alignment, not of
    # the schema: see the gap test below for the contract that always holds.
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


def test_build_episode_delta_ee_drops_deltas_spanning_a_gap(synthetic_export, tmp_path):
    # The wrist camera has no frame at DROPPED_WRIST_TICK, so alignment leaves a
    # hole and the delta across it would report two ticks of motion as one
    # frame's worth. That frame must be dropped, not mislabelled.
    export = load_export(synthetic_export)
    gapless = build_episode(
        export,
        export.sequences[0],
        make_config(synthetic_export, tmp_path, action_space="delta-ee"),
    )
    gapped = build_episode(
        export,
        export.sequences[0],
        make_config(
            synthetic_export,
            tmp_path,
            action_space="delta-ee",
            camera_components=(CAMERA, WRIST_CAMERA),
        ),
    )

    # One tick is unaligned, so GOOD_TICKS - 1 candidate frames; the frame whose
    # delta would span the hole is dropped too.
    assert gapless.n_frames == GOOD_TICKS - 1
    assert gapped.n_frames == GOOD_TICKS - 3
    assert len(gapped.images[CAMERA]) == gapped.n_frames
    assert len(gapped.images[WRIST_CAMERA]) == gapped.n_frames

    # The contract that holds with or without gaps: every surviving pair is one
    # tick of real motion. The gapless run has no holes, so each of its pairs is
    # correct by construction; requiring the gapped pairs to be a subset says
    # the same thing without re-deriving poses from the export.
    reference = {
        (tuple(s), tuple(a)) for s, a in zip(gapless.states, gapless.actions)
    }
    for state, action in zip(gapped.states, gapped.actions):
        assert (tuple(state), tuple(action)) in reference


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


def test_image_size_preserves_aspect_ratio(synthetic_export):
    # Policies disagree on how to square an image (EVO1 squashes to 448, SmolVLA
    # letterboxes to 512), so the converter must only scale, never reshape.
    # Asserted on the loader rather than through convert(): a 16x12 video is
    # small enough to stall the encoder, and _load_image is what feeds both the
    # declared feature shape and the frames.
    export = load_export(synthetic_export)
    path = export.binary_rows(GOOD_SEQ, CAMERA)[0].path
    src_w, src_h = IMAGE_SIZE  # fixture cameras are 32x24

    full = _load_image(path)
    assert full.shape == (src_h, src_w, 3)

    scaled = _load_image(path, image_size=16)
    h, w, c = scaled.shape
    assert (w, h, c) == (16, 12, 3), "longest side pinned to 16, aspect ratio kept"
    assert abs(w / h - src_w / src_h) < 1e-6

    # Larger than the source must not upscale.
    assert _load_image(path, image_size=448).shape == full.shape


def test_downscaled_size_keeps_even_sides_and_never_upscales():
    assert _downscaled_size(1080, 1920, 448) == (252, 448)
    assert _downscaled_size(1280, 720, 448) == (448, 252)
    # An odd result is nudged to even for yuv420, and small inputs are untouched.
    assert all(v % 2 == 0 for v in _downscaled_size(1001, 333, 100))
    assert _downscaled_size(32, 24, 448) == (32, 24)


def test_image_size_rejects_non_positive(synthetic_export, tmp_path):
    with pytest.raises(ValueError, match="image_size must be positive"):
        make_config(synthetic_export, tmp_path, image_size=0)


def _seq(*tags: str) -> Sequence:
    return Sequence(sequence_id="seq", tags=tags, start_at=0.0, end_at=1.0)


def test_sequence_task_strips_prefix():
    assert sequence_task(_seq("session:x", "cmd:open the lid"), "cmd:", None) == "open the lid"


def test_sequence_task_trims_whitespace():
    assert sequence_task(_seq("cmd:  open the lid "), "cmd:", None) == "open the lid"


def test_sequence_task_falls_back_when_no_tag():
    assert sequence_task(_seq("session:x"), "cmd:", "open the box") == "open the box"


def test_sequence_task_skips_when_no_tag_and_no_fallback():
    with pytest.raises(EpisodeSkip, match="no tag with prefix 'cmd:' and no --task fallback"):
        sequence_task(_seq("session:x"), "cmd:", None)


def test_sequence_task_prefix_only_tag_counts_as_absent():
    assert sequence_task(_seq("cmd:", "cmd:   "), "cmd:", "fallback") == "fallback"


def test_sequence_task_skips_on_multiple_tags():
    with pytest.raises(EpisodeSkip, match="2 tags with prefix 'cmd:', expected one"):
        sequence_task(_seq("cmd:a", "cmd:b"), "cmd:", "fallback")


def test_sequence_task_honors_custom_prefix():
    seq = _seq("cmd:ignored", "task:open the lid")
    assert sequence_task(seq, "task:", None) == "open the lid"


def test_config_task_is_optional_and_prefix_defaults(synthetic_export, tmp_path):
    config = make_config(synthetic_export, tmp_path, task=None)
    assert config.task is None
    assert config.task_prefix == "cmd:"


def test_config_rejects_empty_task_prefix(synthetic_export, tmp_path):
    with pytest.raises(ValueError, match="task_prefix"):
        make_config(synthetic_export, tmp_path, task_prefix="")
