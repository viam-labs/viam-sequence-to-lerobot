"""End-effector pose features from Viam ``EndPosition`` payloads.

``observation.state`` is the 9-vector ``[x, y, z, r00, r01, r02, r10, r11,
r12]``: position in millimeters (as captured) followed by the first two rows of
the 3x3 rotation matrix, row-major. A rotation matrix varies smoothly
everywhere in SO(3); no three-number encoding does. That matters here because
the arm holds its tool near-vertical, where the rotation angle sits within
0.03 rad of pi -- exactly where an axis-angle state flips sign under
physically smooth motion.

``action`` is the 6-vector ``[dx, dy, dz, drx, dry, drz]``: the body-frame
motion from one tick to the next, a translation difference plus the relative
rotation ``R_t^-1 . R_t+1`` as an axis-angle vector in radians. Per-tick
rotations are ~0.02 rad, two orders of magnitude clear of the axis-angle
branch cut at pi, so three numbers are both safe and better conditioned than
matrix rows, whose diagonal entries would be pinned at 1.

``state_delta`` and ``state_compose`` are exact inverses: an inference client
composes the policy's delta onto the live ``EndPosition`` exactly the way the
dataset was built, then calls ``orientation_vector`` to get back to a Viam
pose for ``MoveToPosition``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

STATE_NAMES = ["x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz"]

# How close ``o_z`` must be to +/-1 before the orientation vector counts as
# being on a pole, where its longitude is undefined and the ZYZ euler split is
# gimbal locked. Mirrors defaultAngleEpsilon in rdk spatialmath and must keep
# matching it: at 1e-7 instead, poses within ~0.8 deg of vertical decode to a
# rotation up to 63 deg away from the one rdk encoded.
_POLE_EPSILON = 1e-4


def pose_rotation(pose: dict) -> Rotation:
    """Rotation described by a Viam pose's orientation vector.

    Mirrors rdk ``spatialmath/orientationVector.go``: the orientation vector is
    the tool's z axis, decoded as ZYZ euler angles ``(lon=atan2(o_y, o_x),
    lat=acos(o_z), theta)``, with ``theta`` in degrees as the protobuf ``Pose``
    carries it.

    Raises:
        ValueError: If the orientation vector has zero length.
    """
    o = np.array([pose["o_x"], pose["o_y"], pose["o_z"]], dtype=np.float64)
    norm = np.linalg.norm(o)
    if norm == 0.0:
        raise ValueError("orientation vector has zero length (o_x, o_y, o_z all 0)")
    o /= norm
    lat = np.arccos(np.clip(o[2], -1.0, 1.0))
    lon = np.arctan2(o[1], o[0]) if 1.0 - abs(o[2]) > _POLE_EPSILON else 0.0
    return Rotation.from_euler("ZYZ", [lon, lat, np.deg2rad(pose["theta"])])


def orientation_vector(rotation: Rotation) -> dict:
    """Viam orientation-vector fields for a rotation; inverse of pose_rotation.

    Returns the tool's z axis as ``o_x``, ``o_y``, ``o_z`` (the third column of
    the rotation matrix) plus ``theta`` in degrees, ready to drop into a
    ``Pose`` for ``MoveToPosition``. On a pole the ``(lon, theta)`` split is
    gimbal locked, so the fields need not match the ones originally captured;
    the rotation they describe is the same either way.
    """
    o_x, o_y, o_z = rotation.apply([0.0, 0.0, 1.0])
    theta = rotation.as_euler("ZYZ")[2]
    return {
        "o_x": float(o_x),
        "o_y": float(o_y),
        "o_z": float(o_z),
        "theta": float(np.degrees(theta)),
    }


def pose_state(payload: dict) -> np.ndarray:
    """Build the 9-dim ``observation.state`` from an ``EndPosition`` payload."""
    pose = payload["pose"]
    rotation = pose_rotation(pose)
    return np.concatenate(
        [
            [pose["x"], pose["y"], pose["z"]],
            rotation.as_matrix()[:2, :].reshape(6),
        ]
    )


def state_rotation(state: np.ndarray) -> Rotation:
    """Rotation held in a state's two matrix rows, recovered by Gram-Schmidt.

    Any six numbers yield a valid rotation, so a policy's raw output needs no
    orthogonality constraint -- only the degenerate cases below are rejected.

    Raises:
        ValueError: If the first row is zero, or the second is parallel to it.
    """
    a1 = np.asarray(state[3:6], dtype=np.float64)
    a2 = np.asarray(state[6:9], dtype=np.float64)
    n1 = np.linalg.norm(a1)
    if n1 == 0.0:
        raise ValueError("state[3:6] has zero length; no rotation to recover")
    b1 = a1 / n1
    b2 = a2 - (b1 @ a2) * b1
    n2 = np.linalg.norm(b2)
    if n2 == 0.0:
        raise ValueError("state[6:9] is parallel to state[3:6]; no rotation to recover")
    b2 /= n2
    return Rotation.from_matrix(np.stack([b1, b2, np.cross(b1, b2)]))


def state_delta(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Body-frame motion from ``current`` to ``target`` as the 6-dim action."""
    relative = state_rotation(current).inv() * state_rotation(target)
    translation = np.asarray(target[:3], dtype=np.float64) - np.asarray(
        current[:3], dtype=np.float64
    )
    return np.concatenate([translation, relative.as_rotvec()])


def state_compose(current: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a ``state_delta`` result to ``current``; inverse of ``state_delta``."""
    rotation = state_rotation(current) * Rotation.from_rotvec(
        np.asarray(delta[3:6], dtype=np.float64)
    )
    position = np.asarray(current[:3], dtype=np.float64) + np.asarray(
        delta[:3], dtype=np.float64
    )
    return np.concatenate([position, rotation.as_matrix()[:2, :].reshape(6)])
