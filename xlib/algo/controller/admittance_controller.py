import numpy as np
from scipy.spatial.transform import Rotation as R

from ..utils.transforms import applyDeltaPose6d, calcPose6dError, velTransform


class AdmittanceController(object):
    def __init__(self, M, D, K, threshold_high, threshold_low):
        self.M = M
        self.D = D
        self.K = K
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.force_flag = False
        self.torque_flag = True
        self.x_dot = np.zeros(6)

    def update_params(self, M=None, D=None, K=None):
        if M is not None:
            self.M = M
        if D is not None:
            self.D = D
        if K is not None:
            self.K = K

    def update(self, x, x_exp, dt, x_dot=None, f_ext=np.zeros(6)):
        f = np.zeros(6)
        force_norm = np.linalg.norm(f_ext[:3])
        torque_norm = np.linalg.norm(f_ext[3:])
        if force_norm > self.threshold_high[0]:
            self.force_flag = True
        if force_norm < self.threshold_low[0]:
            self.force_flag = False
        if torque_norm > self.threshold_high[1]:
            self.torque_flag = True
        if torque_norm < self.threshold_low[1]:
            self.torque_flag = False
        if self.force_flag:
            f[:3] = f_ext[:3] - self.threshold_low[0] * f_ext[:3] / force_norm
        if self.torque_flag:
            f[3:] = f_ext[3:] - self.threshold_low[1] * f_ext[3:] / torque_norm

        if x_dot is not None:
            self.x_dot = x_dot
        pose_error = calcPose6dError(x_exp, x)
        pose_error = velTransform(pose_error, R.from_rotvec(x_exp[3:]).as_matrix())
        print("pose error", pose_error)
        x_ddot = np.linalg.inv(self.M) @ (f - self.D @ self.x_dot - self.K @ pose_error)

        self.x_dot += x_ddot * dt
        x_rot = R.from_rotvec(x[3:]).as_matrix()
        x_dot_in_tcp = velTransform(self.x_dot, np.linalg.inv(x_rot))
        x = applyDeltaPose6d(x, x_dot_in_tcp * dt)

        return x, self.x_dot
