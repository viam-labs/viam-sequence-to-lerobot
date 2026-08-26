from __future__ import annotations

import numpy as np
import pytest

from viam_sequence_to_lerobot.pose import POSE_NAMES, pose_compose, pose_delta, pose_vector


def payload(x=0.0, y=0.0, z=0.0, o_x=0.0, o_y=0.0, o_z=1.0, theta=0.0) -> dict:
    return {
        "pose": {"x": x, "y": y, "z": z, "o_x": o_x, "o_y": o_y, "o_z": o_z, "theta": theta}
    }


def test_pose_names():
    assert POSE_NAMES == ["x", "y", "z", "rx", "ry", "rz"]


def test_pose_vector_identity():
    vec = pose_vector(payload(x=1.0, y=2.0, z=3.0))
    np.testing.assert_allclose(vec, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_pose_vector_theta_rotates_about_z():
    vec = pose_vector(payload(theta=90.0))
    np.testing.assert_allclose(vec[3:], [0.0, 0.0, np.pi / 2], atol=1e-9)


def test_pose_vector_z_axis_tilted_onto_x():
    # Orientation vector (1,0,0) points the tool Z axis along +X: a 90° pitch about Y.
    vec = pose_vector(payload(o_x=1.0, o_z=0.0))
    np.testing.assert_allclose(vec[3:], [0.0, np.pi / 2, 0.0], atol=1e-9)


def test_pose_delta_translation():
    a = pose_vector(payload(x=1.0, y=2.0, z=3.0))
    b = pose_vector(payload(x=2.0, y=1.0, z=6.0))
    np.testing.assert_allclose(pose_delta(a, b)[:3], [1.0, -1.0, 3.0], atol=1e-9)


def test_pose_delta_same_axis_rotation():
    a = pose_vector(payload(theta=30.0))
    b = pose_vector(payload(theta=50.0))
    np.testing.assert_allclose(pose_delta(a, b)[3:], [0.0, 0.0, np.deg2rad(20.0)], atol=1e-9)


def test_pose_delta_identity_when_equal():
    a = pose_vector(payload(x=5.0, o_x=0.3, o_y=0.4, o_z=0.5, theta=17.0))
    np.testing.assert_allclose(pose_delta(a, a), np.zeros(6), atol=1e-9)


def test_pose_compose_inverts_delta():
    rng = np.random.default_rng(7)
    for _ in range(10):
        vecs = []
        for _ in range(2):
            o = rng.normal(size=3)
            o /= np.linalg.norm(o)
            x, y, z = rng.normal(scale=100.0, size=3)
            vecs.append(
                pose_vector(
                    payload(
                        x=x, y=y, z=z,
                        o_x=o[0], o_y=o[1], o_z=o[2],
                        theta=rng.uniform(-170.0, 170.0),
                    )
                )
            )
        a, b = vecs
        np.testing.assert_allclose(pose_compose(a, pose_delta(a, b)), b, atol=1e-9)


def test_pose_vector_matches_real_export_sample():
    # Real EndPosition payload from the workshop export; guards the OV convention
    # (ZYZ euler: lon=atan2(o_y,o_x), lat=acos(o_z), theta degrees) end to end.
    vec = pose_vector(
        {
            "pose": {
                "o_x": 0.03042069263169411,
                "o_y": -0.02081466215692586,
                "o_z": -0.9993204347450825,
                "theta": -35.48005255994949,
                "x": 356.9206023013329,
                "y": -87.52599428622149,
                "z": 395.31523393430484,
            }
        }
    )
    assert vec.shape == (6,)
    np.testing.assert_allclose(vec[:3], [356.9206023013329, -87.52599428622149, 395.31523393430484])
    # Tool Z axis nearly inverted (pointing down): rotation angle close to pi.
    assert np.linalg.norm(vec[3:]) == pytest.approx(np.pi, abs=0.15)
