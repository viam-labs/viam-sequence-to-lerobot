"""Differential test: pose.py's scipy OV decode vs viam.spatialmath.

The converter decodes orientation vectors with scipy; inference clients use
``viam.spatialmath`` (Rust FFI, golden-vector parity-tested against Go rdk).
Two implementations of one convention only stay interchangeable if they agree
to machine precision, which is what this pins.
"""

from __future__ import annotations

import numpy as np
import pytest

viam_spatialmath = pytest.importorskip("viam.spatialmath")

from viam_sequence_to_lerobot.pose import pose_rotation

OrientationVector = viam_spatialmath.OrientationVector


def sdk_matrix(o: np.ndarray, theta_deg: float) -> np.ndarray:
    # SDK spatialmath takes theta in radians; the wire Pose (and our payloads)
    # carry degrees.
    ov = OrientationVector(o[0], o[1], o[2], np.deg2rad(theta_deg))
    return np.array(ov.to_quaternion().to_rotation_matrix().elements).reshape(3, 3)


def our_matrix(o: np.ndarray, theta_deg: float) -> np.ndarray:
    pose = {"o_x": o[0], "o_y": o[1], "o_z": o[2], "theta": theta_deg}
    return pose_rotation(pose).as_matrix()


def assert_parity(o: np.ndarray, theta_deg: float) -> None:
    np.testing.assert_allclose(
        our_matrix(o, theta_deg),
        sdk_matrix(o, theta_deg),
        atol=1e-9,
        err_msg=f"disagreement at o={o.tolist()}, theta={theta_deg}",
    )


def test_parity_on_random_orientation_vectors():
    rng = np.random.default_rng(42)
    for _ in range(2000):
        o = rng.normal(size=3)
        o /= np.linalg.norm(o)
        assert_parity(o, rng.uniform(-180.0, 180.0))


def pole_vector(rng, gap: float, sign: float) -> np.ndarray:
    azimuth = rng.uniform(-np.pi, np.pi)
    tilt = np.arccos(np.clip(1.0 - gap, -1.0, 1.0))
    return np.array(
        [
            np.sin(tilt) * np.cos(azimuth),
            np.sin(tilt) * np.sin(azimuth),
            sign * np.cos(tilt),
        ]
    )


def test_parity_near_poles():
    # The pole epsilon (1e-4 on 1-|o_z|) is where conventions diverge first;
    # sample both sides of it. The south-pole band inside the epsilon is
    # excluded: rust-utils diverges from Go rdk there (see canary test below).
    rng = np.random.default_rng(7)
    for gap in (0.0, 1e-6, 5e-5, 2e-4, 1e-3, 1e-2):
        for sign in (1.0, -1.0):
            if sign < 0 and gap <= 1e-4:
                continue
            for _ in range(20):
                assert_parity(pole_vector(rng, gap, sign), rng.uniform(-180.0, 180.0))


@pytest.mark.xfail(
    strict=True,
    reason="rust-utils bug: OV decode pins lon=0 only at the north pole "
    "(src/spatialmath/utils.rs `1.0 - val > ANGLE_ACCEPTANCE`, no abs), so "
    "within 1e-4 of straight-down it disagrees with Go rdk's "
    "`1 - math.Abs(ov.OZ)`. pose.py follows Go. When this XPASSes, upstream "
    "fixed it: fold this band back into test_parity_near_poles.",
)
def test_parity_inside_south_pole_band_upstream_bug_canary():
    rng = np.random.default_rng(11)
    for gap in (1e-6, 5e-5):
        for _ in range(20):
            assert_parity(pole_vector(rng, gap, -1.0), rng.uniform(-180.0, 180.0))


def test_parity_on_real_export_payload():
    o = np.array([0.03042069263169411, -0.02081466215692586, -0.9993204347450825])
    assert_parity(o, -35.48005255994949)
