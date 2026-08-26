"""End-effector pose vectors from Viam ``EndPosition`` payloads.

A pose is a 6-vector ``[x, y, z, rx, ry, rz]``: position in millimeters (as
captured) and orientation as an axis-angle rotation vector in radians. Deltas
are body-frame: ``pose_delta(a, b)`` is the motion that takes ``a`` to ``b``,
and ``pose_compose(a, delta)`` applies it — an inference client composes the
policy's delta onto the live ``EndPosition`` the same way.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

POSE_NAMES = ["x", "y", "z", "rx", "ry", "rz"]

# Rotation angles about z below which the orientation vector is on a pole and
# its longitude is undefined; mirrors defaultAngleEpsilon in rdk spatialmath.
_POLE_EPSILON = 1e-7


def _rotation(pose: dict) -> Rotation:
    # Viam orientation vector -> rotation, per rdk spatialmath
    # orientationVector.go: ZYZ euler (lon=atan2(o_y,o_x), lat=acos(o_z), theta).
    o = np.array([pose["o_x"], pose["o_y"], pose["o_z"]], dtype=np.float64)
    o /= np.linalg.norm(o)
    lat = np.arccos(np.clip(o[2], -1.0, 1.0))
    lon = np.arctan2(o[1], o[0]) if 1.0 - abs(o[2]) > _POLE_EPSILON else 0.0
    return Rotation.from_euler("ZYZ", [lon, lat, np.deg2rad(pose["theta"])])


def pose_vector(payload: dict) -> np.ndarray:
    """Extract ``[x, y, z, rx, ry, rz]`` from an ``EndPosition`` payload."""
    pose = payload["pose"]
    return np.concatenate(
        [
            [pose["x"], pose["y"], pose["z"]],
            _rotation(pose).as_rotvec(),
        ]
    )


def pose_delta(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Motion from ``current`` to ``target``: xyz difference, body-frame rotvec."""
    rotation = Rotation.from_rotvec(current[3:]).inv() * Rotation.from_rotvec(target[3:])
    return np.concatenate([target[:3] - current[:3], rotation.as_rotvec()])


def pose_compose(current: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a ``pose_delta`` result to ``current``; inverse of ``pose_delta``."""
    rotation = Rotation.from_rotvec(current[3:]) * Rotation.from_rotvec(delta[3:])
    return np.concatenate([current[:3] + delta[:3], rotation.as_rotvec()])
