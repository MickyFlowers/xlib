import numpy as np
from scipy.spatial.transform import Rotation as R

from .transforms import calcNormVecFromPoint, matrixToPose6d


def sample_point_on_sphere(
    center: np.ndarray,
    radius: float,
    yaw_lower: float = 0.0,
    yaw_upper: float = np.pi * 2,
    pitch_lower: float = -np.pi / 2,
    pitch_upper: float = np.pi / 2,
) -> np.ndarray:
    """Sample a point on the sphere with center and radius

    Args:
        center (np.ndarray): center of the sphere
        radius (float): radius of the sphere
    Return:
        np.ndarray: point on the sphere
    """
    if yaw_lower > yaw_upper:
        yaw_lower, yaw_upper = yaw_upper, yaw_lower
    if pitch_lower > pitch_upper:
        pitch_lower, pitch_upper = pitch_upper, pitch_lower

    u1 = np.random.uniform(pitch_lower, pitch_upper)
    u2 = np.random.uniform(yaw_lower, yaw_upper)

    x = center[0] + radius * np.cos(u1) * np.cos(u2)
    y = center[1] + radius * np.cos(u1) * np.sin(u2)
    z = center[2] + radius * np.sin(u1)
    return np.array([x, y, z]), u1, u2


def sample_cordinate_on_sphere(
    center: np.ndarray,
    radius: float,
    yaw_lower: float = 0.0,
    yaw_upper: float = np.pi * 2,
    pitch_lower: float = -np.pi / 2,
    pitch_upper: float = np.pi / 2,
) -> np.ndarray:
    """Sample a point on the sphere with center and radius

    Args:
        center (np.ndarray): center of the sphere
        radius (float): radius of the sphere
    Return:
        np.ndarray: point on the sphere
    """
    sample_point, _, _ = sample_point_on_sphere(
        center, radius, yaw_lower, yaw_upper, pitch_lower, pitch_upper
    )
    z_axis = calcNormVecFromPoint(sample_point, center)

    arbitrary_vector = np.array([0, 0, 1])

    def are_vectors_parallel(v1, v2, tol=1e-10):
        cross_product = np.cross(v1, v2)
        return np.all(np.abs(cross_product) < tol)

    if are_vectors_parallel(z_axis, arbitrary_vector):
        arbitrary_vector = np.array([0, 1, 0])
    x_axis = np.cross(z_axis, arbitrary_vector)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    trans_matrix = np.eye(4)
    trans_matrix[:3, :3] = rotation_matrix
    trans_matrix[:3, 3] = sample_point
    return trans_matrix


def sample_disturbance(
    lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:

    """Add disturbance to the pose
    Args:
        pose (np.ndarray): pose matrix
        trans_lower (np.ndarray): lower bound of translation disturbance
        trans_upper (np.ndarray): upper bound of translation disturbance
        rot_lower (np.ndarray): lower bound of rotation disturbance, xyz euler angle
        rot_upper (np.ndarray): upper bound of rotation disturbance, xyz euler angle
    """
    rot_lower = lower[3:6]
    rot_upper = upper[3:6]
    trans_lower = lower[0:3]
    trans_upper = upper[0:3]
    rot_disturbance = np.random.uniform(rot_lower, rot_upper)
    disturbance_rot_matrix = R.from_euler("xyz", rot_disturbance).as_matrix()
    trans_disturbance = np.random.uniform(trans_lower, trans_upper)

    pose_disturbed = np.eye(4)
    pose_disturbed[:3, :3] = disturbance_rot_matrix
    pose_disturbed[:3, 3] = trans_disturbance

    return matrixToPose6d(pose_disturbed)
