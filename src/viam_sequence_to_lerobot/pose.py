"""End-effector pose features from Viam ``EndPosition`` payloads.

``observation.state`` is ``[x, y, z, r00, r01, r02, r10, r11, r12]``:
millimeters as captured, then the first two ROWS of the 3x3 rotation matrix.
Rows spend six numbers on rotation because no three-number encoding is
continuous everywhere, and this arm works where that bites.

``action`` is ``[dx, dy, dz, drx, dry, drz]``: translation difference plus the
body-frame rotation ``R_t^-1 . R_t+1`` as an axis-angle vector in radians.

``state_delta`` and ``state_compose`` are exact inverses; an inference client
composes the policy's delta onto the live pose with ``state_compose``, then
``orientation_vector`` to get back to a Viam pose for ``MoveToPosition``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from viam.spatialmath import OrientationVector, RotationMatrix

STATE_NAMES = ["x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz"]


def pose_rotation(pose: dict) -> Rotation:
    """Rotation described by a Viam pose's orientation vector, via the SDK.

    Goes through the SDK's rotation matrix, not its quaternion, so no component
    order is written by hand: scipy orders a quaternion ``(x, y, z, w)`` and the
    SDK ``(w, i, j, k)``. ``theta`` arrives in degrees, as the protobuf ``Pose``
    carries it. The SDK pins the orientation vector's longitude only near
    ``o_z = +1`` where rdk pins both poles -- see the canaries in
    ``tests/test_pose.py``.

    Raises:
        ValueError: If the orientation vector has zero length.
    """
    if pose["o_x"] == 0.0 and pose["o_y"] == 0.0 and pose["o_z"] == 0.0:
        raise ValueError("orientation vector has zero length (o_x, o_y, o_z all 0)")
    quaternion = OrientationVector(
        pose["o_x"], pose["o_y"], pose["o_z"], np.deg2rad(pose["theta"])
    ).to_quaternion()
    elements = np.asarray(quaternion.to_rotation_matrix().elements, dtype=np.float64)
    return Rotation.from_matrix(elements.reshape(3, 3))


def orientation_vector(rotation: Rotation) -> dict:
    """Viam orientation-vector fields for a rotation; inverse of pose_rotation.

    ``theta`` comes back in degrees, ready for a ``Pose``. On a pole the
    longitude/theta split is gimbal locked, so the fields need not match the
    ones originally captured; the rotation is the same either way.
    """
    matrix = RotationMatrix(rotation.as_matrix().reshape(9).tolist())
    vector = matrix.to_quaternion().to_orientation_vector()
    return {
        "o_x": float(vector.o_x),
        "o_y": float(vector.o_y),
        "o_z": float(vector.o_z),
        "theta": float(np.degrees(vector.theta)),
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
    """Rotation held in a state's two matrix rows, via Gram-Schmidt.

    Any six numbers yield a valid rotation, so a policy's raw output needs no
    orthogonality constraint.

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
