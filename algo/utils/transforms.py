import numpy as np
from scipy.spatial.transform import Rotation as R


def linkVelTransform(vel, trans_matrix):
    """transform link velocity from frame A to frame B

    Args:
        vel (_type_): frame A velocity
        trans_matrix (_type_): transformation matrix from frame A to frame B

    Returns:
        _type_: frame B velocity
    """
    angular_vel = vel[3:]
    linear_vel = vel[:3]
    angular_transform_vel = trans_matrix[:3, :3] @ angular_vel
    linear_transform_vel = trans_matrix[:3, :3] @ linear_vel - np.cross(
        trans_matrix[:3, 3], angular_transform_vel
    )
    return np.concatenate((linear_transform_vel, angular_transform_vel))


def velTransform(vel, rot_matrix):
    """transform velocity from frame A to frame B

    Args:
        vel (_type_): velocity in frame A
        rot_matrix (_type_): rotation matrix from frame A to frame B

    Returns:
        _type_: velocity in frame B
    """
    trans_matrix = np.eye(6)
    trans_matrix[:3, :3] = rot_matrix
    trans_matrix[3:, 3:] = rot_matrix
    return trans_matrix @ vel


def vecToMatrix(rvec, tvec):
    matrix = np.eye(4)
    rot_matrix = R.from_rotvec(rvec).as_matrix()
    matrix[:3, :3] = rot_matrix
    matrix[:3, 3] = tvec
    return matrix

def calcNormVecFromPoint(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    vec = target - origin
    vec = vec / np.linalg.norm(vec)
    return vec