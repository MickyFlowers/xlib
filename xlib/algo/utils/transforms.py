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
    linear_transform_vel = trans_matrix[:3, :3] @ linear_vel - np.cross(trans_matrix[:3, 3], angular_transform_vel)
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


def matrixToVec(matrix: np.ndarray) -> np.ndarray:
    rvec = R.from_matrix(matrix[:3, :3]).as_rotvec()
    tvec = matrix[:3, 3]
    return rvec.reshape((3, 1)), tvec.reshape((3, 1))


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




def calcPose6dError(pose1, pose2):
    pose1_mtx = np.eye(4)
    pose1_mtx[:3, :3] = R.from_rotvec(pose1[3:]).as_matrix()
    pose1_mtx[:3, 3] = pose1[:3]
    pose2_mtx = np.eye(4)
    pose2_mtx[:3, :3] = R.from_rotvec(pose2[3:]).as_matrix()
    pose2_mtx[:3, 3] = pose2[:3]
    delta_mtx = np.linalg.inv(pose1_mtx) @ pose2_mtx
    delta_pos = delta_mtx[:3, 3]
    delta_rot_vec = R.from_matrix(delta_mtx[:3, :3]).as_rotvec()
    return np.concatenate((delta_pos, delta_rot_vec))


def applyDeltaPose6d(pose, delta_pose):
    pose_mtx = np.eye(4)
    pose_mtx[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    pose_mtx[:3, 3] = pose[:3]
    delta_mtx = np.eye(4)
    delta_mtx[:3, :3] = R.from_rotvec(delta_pose[3:]).as_matrix()
    delta_mtx[:3, 3] = delta_pose[:3]
    new_pose_mtx = pose_mtx @ delta_mtx
    new_pos = new_pose_mtx[:3, 3]
    new_rot_vec = R.from_matrix(new_pose_mtx[:3, :3]).as_rotvec()
    return np.concatenate((new_pos, new_rot_vec))
