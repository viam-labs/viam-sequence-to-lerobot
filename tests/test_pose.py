from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viam_sequence_to_lerobot.pose import (
    ACTION_NAMES,
    STATE_NAMES,
    orientation_vector,
    pose_rotation,
    pose_state,
    state_compose,
    state_delta,
    state_rotation,
)

# Real EndPosition payload from the workshop export: the tool points nearly
# straight down, 2.1 deg off vertical, which puts its rotation angle within
# 0.03 rad of pi. Every near-pole / branch-cut test below is anchored here
# because that is where this arm actually spends its time.
WORKSHOP_POSE = {
    "o_x": 0.03042069263169411,
    "o_y": -0.02081466215692586,
    "o_z": -0.9993204347450825,
    "theta": -35.48005255994949,
    "x": 356.9206023013329,
    "y": -87.52599428622149,
    "z": 395.31523393430484,
}


def payload(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0) -> dict:
    return {
        "pose": {"x": x, "y": y, "z": z, "o_x": o_x, "o_y": o_y, "o_z": o_z, "theta": theta}
    }


def payload_for(rotation: Rotation, position=(0.0, 0.0, 0.0)) -> dict:
    """An EndPosition payload describing ``rotation``, via Viam's OV fields."""
    x, y, z = position
    return {"pose": {"x": x, "y": y, "z": z, **orientation_vector(rotation)}}


def test_feature_names():
    assert STATE_NAMES == ["x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12"]
    assert ACTION_NAMES == ["dx", "dy", "dz", "drx", "dry", "drz"]


def test_pose_state_identity():
    state = pose_state(payload(x=1.0, y=2.0, z=3.0))
    assert state.shape == (9,)
    # Identity rotation: first two rows of the identity matrix.
    np.testing.assert_allclose(state, [1.0, 2.0, 3.0, 1, 0, 0, 0, 1, 0], atol=1e-12)


def test_pose_state_rows_are_the_rotation_matrix():
    state = pose_state({"pose": WORKSHOP_POSE})
    expected = pose_rotation(WORKSHOP_POSE).as_matrix()
    np.testing.assert_allclose(state[3:9], expected[:2, :].reshape(6), atol=1e-12)
    np.testing.assert_allclose(
        state[:3], [WORKSHOP_POSE["x"], WORKSHOP_POSE["y"], WORKSHOP_POSE["z"]]
    )


def test_orientation_vector_is_the_third_matrix_column():
    # Viam's defining property: the OV is where the tool's z axis points.
    rng = np.random.default_rng(11)
    for _ in range(50):
        o = rng.normal(size=3)
        o /= np.linalg.norm(o)
        rotation = pose_rotation(
            {"o_x": o[0], "o_y": o[1], "o_z": o[2], "theta": rng.uniform(-180, 180)}
        )
        np.testing.assert_allclose(rotation.as_matrix()[:, 2], o, atol=1e-12)


def test_pole_epsilon_matches_rdk():
    # rdk's defaultAngleEpsilon is 1e-4: inside that band it zeroes the
    # longitude. A smaller epsilon here decodes near-vertical poses to a
    # rotation up to 63 deg away from the one rdk encoded.
    gap = 1e-5  # 1 - |o_z|, i.e. ~0.26 deg from straight down
    o_z = -(1 - gap)
    radius = np.sqrt(1 - o_z**2)
    pose = {
        "o_x": radius * np.cos(1.1),
        "o_y": radius * np.sin(1.1),
        "o_z": o_z,
        "theta": -35.48,
    }
    rdk = Rotation.from_euler("ZYZ", [0.0, np.arccos(o_z), np.deg2rad(pose["theta"])])
    assert np.degrees((pose_rotation(pose).inv() * rdk).magnitude()) < 1e-9


def test_pose_rotation_rejects_zero_orientation_vector():
    with pytest.raises(ValueError, match="zero length"):
        pose_state(payload(o_z=0.0))


def test_state_rotation_inverts_pose_state():
    state = pose_state({"pose": WORKSHOP_POSE})
    recovered = state_rotation(state)
    expected = pose_rotation(WORKSHOP_POSE)
    assert (recovered.inv() * expected).magnitude() < 1e-12


def test_state_rotation_orthonormalizes_arbitrary_rows():
    # A policy emits six unconstrained numbers; Gram-Schmidt must still yield a
    # valid rotation whose first row follows the first input row.
    state = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0])
    matrix = state_rotation(state).as_matrix()
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(matrix[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_state_rotation_rejects_degenerate_rows():
    with pytest.raises(ValueError, match=r"state\[3:6\]"):
        state_rotation(np.zeros(9))
    with pytest.raises(ValueError, match=r"state\[6:9\]"):
        state_rotation(np.array([0, 0, 0, 1.0, 0, 0, 2.0, 0, 0]))


def test_state_delta_translation():
    a = pose_state(payload(x=1.0, y=2.0, z=3.0))
    b = pose_state(payload(x=2.0, y=1.0, z=6.0))
    np.testing.assert_allclose(state_delta(a, b), [1.0, -1.0, 3.0, 0, 0, 0], atol=1e-12)


def test_state_delta_same_axis_rotation():
    a = pose_state(payload(theta=30.0))
    b = pose_state(payload(theta=50.0))
    np.testing.assert_allclose(
        state_delta(a, b)[3:], [0.0, 0.0, np.deg2rad(20.0)], atol=1e-12
    )


def test_state_delta_identity_when_equal():
    a = pose_state(payload(x=5.0, o_x=0.3, o_y=0.4, o_z=0.5, theta=17.0))
    np.testing.assert_allclose(state_delta(a, a), np.zeros(6), atol=1e-12)


def test_state_compose_inverts_delta():
    rng = np.random.default_rng(7)
    for _ in range(20):
        states = []
        for _ in range(2):
            o = rng.normal(size=3)
            o /= np.linalg.norm(o)
            x, y, z = rng.normal(scale=100.0, size=3)
            states.append(
                pose_state(
                    payload(
                        x=x, y=y, z=z,
                        o_x=o[0], o_y=o[1], o_z=o[2],
                        theta=rng.uniform(-170.0, 170.0),
                    )
                )
            )
        a, b = states
        np.testing.assert_allclose(state_compose(a, state_delta(a, b)), b, atol=1e-9)


def test_state_is_continuous_across_the_axis_angle_branch_cut():
    """Smooth motion must produce smooth state, near the pi branch cut.

    The workshop pose sits ~0.03 rad below a rotation angle of pi. An
    axis-angle state flips sign there: a 0.02 rad step produced a jump of
    6.26 in the state vector, while the physical motion was 0.02. Matrix rows
    have no branch cut, so every step must stay proportional to the motion.

    A compose/decompose round trip cannot catch this -- Rotation.from_rotvec
    is branch-agnostic, so the invariant held on exactly the frames that
    flipped. Only consecutive-frame continuity detects it.
    """
    start = pose_rotation(WORKSHOP_POSE)
    assert np.pi - start.magnitude() < 0.05, "fixture must sit near the branch cut"
    axis = start.as_rotvec() / np.linalg.norm(start.as_rotvec())
    step = Rotation.from_rotvec(0.02 * axis)

    rotation = start
    states, angles = [], []
    for _ in range(6):
        states.append(pose_state(payload_for(rotation)))
        angles.append(rotation.magnitude())
        rotation = rotation * step
    jumps = np.linalg.norm(np.diff(np.stack(states), axis=0), axis=1)

    # 0.02 rad of rotation moves two unit-norm matrix rows by at most ~0.03.
    assert jumps.max() < 0.05, f"state jumped by {jumps.max():.4f} on smooth motion"
    # The sweep really does reach the cut, so the assertion above is live. The
    # angle rises to pi and then folds back, which is exactly the fold that
    # flips an axis-angle vector's sign.
    assert max(angles) > np.pi - 0.015
    assert angles[-1] < max(angles)


def test_orientation_vector_round_trips_for_move_to_position():
    # The inference client turns a composed rotation back into Viam OV fields.
    # On a pole the (lon, theta) split is gimbal locked, so require the
    # rotation to match rather than the individual fields.
    rng = np.random.default_rng(5)
    for _ in range(50):
        rotation = Rotation.random(random_state=int(rng.integers(1 << 31)))
        fields = orientation_vector(rotation)
        assert (pose_rotation(fields).inv() * rotation).magnitude() < 1e-9


def test_orientation_vector_recovers_workshop_fields():
    fields = orientation_vector(pose_rotation(WORKSHOP_POSE))
    for key in ("o_x", "o_y", "o_z", "theta"):
        assert fields[key] == pytest.approx(WORKSHOP_POSE[key], abs=1e-9)


def test_pose_state_matches_real_export_sample():
    state = pose_state({"pose": WORKSHOP_POSE})
    assert state.shape == (9,)
    np.testing.assert_allclose(
        state[:3], [356.9206023013329, -87.52599428622149, 395.31523393430484]
    )
    np.testing.assert_allclose(
        state[3:9],
        [-0.999359, -0.018856, 0.030421, -0.019494, 0.999593, -0.020815],
        atol=1e-6,
    )
