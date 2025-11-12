import numpy as np
from scipy.spatial.transform import Rotation as R


def linkVelTransform(vel, trans_vec):
    """transform link velocity from frame A to frame B

    Args:
        vel (_type_): frame A velocity`
        trans_vec (_type_): translation vector from frame A to frame B

    Returns:
        _type_: frame B velocity
    """
    angular_vel = vel[3:]
    linear_vel = vel[:3]
    trans_matrix = pose6dToMatrix(trans_vec)
    angular_transform_vel = trans_matrix[:3, :3] @ angular_vel
    linear_transform_vel = trans_matrix[:3, :3] @ linear_vel - np.cross(trans_matrix[:3, 3], angular_transform_vel)
    return np.concatenate((linear_transform_vel, angular_transform_vel))


def velTransform(vel, rot_vec):
    """transform velocity from frame A to frame B

    Args:
        vel (_type_): velocity in frame A
        rot_vec (_type_): rotation vec from frame A to frame B

    Returns:
        _type_: velocity in frame B
    """
    rot_matrix = R.from_rotvec(rot_vec).as_matrix()
    trans_matrix = np.eye(6)
    trans_matrix[:3, :3] = rot_matrix
    trans_matrix[3:, 3:] = rot_matrix
    return trans_matrix @ vel


def matrixToVecCv2(matrix: np.ndarray) -> np.ndarray:
    rvec = R.from_matrix(matrix[:3, :3]).as_rotvec()
    tvec = matrix[:3, 3]
    return rvec.reshape((3, 1)), tvec.reshape((3, 1))


def vecCv2ToMatrix(rvec, tvec):
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


def pose6dToMatrix(pose):
    matrix = np.eye(4)
    rot_matrix = R.from_rotvec(pose[3:]).as_matrix()
    matrix[:3, :3] = rot_matrix
    matrix[:3, 3] = pose[:3]
    return matrix


def invPose6d(pose):
    matrix = pose6dToMatrix(pose)
    inv_matrix = np.linalg.inv(matrix)
    return matrixToPose6d(inv_matrix)


def matrixToPose6d(matrix):
    rot_vec = R.from_matrix(matrix[:3, :3]).as_rotvec()
    tvec = matrix[:3, 3]
    return np.concatenate((tvec, rot_vec))
