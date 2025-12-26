import numpy as np

from ..utils.transforms import applyDeltaPose6d, calcPose6dError


class AdmittanceController(object):
    def __init__(self, M, D, K, threshold_high, threshold_low, K_min_ratio=0.2):
        """
        Args:
            M, D, K: 导纳参数
            threshold_high: 力激活阈值 [force, torque]
            threshold_low: 力死区阈值 [force, torque]
            K_min_ratio: 接触时K的最小比例 (0~1)，用于自适应刚度
        """
        if not isinstance(M, np.ndarray):
            M = np.array(M)
            D = np.array(D)
            K = np.array(K)
            threshold_high = np.array(threshold_high)
            threshold_low = np.array(threshold_low)

        if M.ndim == 1:
            M = np.diag(M)
            D = np.diag(D)
            K = np.diag(K)
        self.M = M
        self.D = D
        self.K = K
        self.K_base = K.copy()
        self.K_min_ratio = K_min_ratio
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.force_flag = False
        self.torque_flag = False
        self.x_dot = np.zeros(6)
        self.x = np.zeros(6)

    def reset(self):
        self.force_flag = False
        self.torque_flag = False
        self.x_dot = np.zeros(6)
        self.x = np.zeros(6)
        self.K = self.K_base.copy()

    def update_params(self, M=None, D=None, K=None):
        if M is not None:
            self.M = M
        if D is not None:
            self.D = D
        if K is not None:
            self.K = K

    def _apply_deadzone(self, f_ext):
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
        if self.force_flag and force_norm > 0:
            f[:3] = f_ext[:3] - self.threshold_low[0] * f_ext[:3] / force_norm
        if self.torque_flag and torque_norm > 0:
            f[3:] = f_ext[3:] - self.threshold_low[1] * f_ext[3:] / torque_norm
        return f

    def _compute_adaptive_K(self, f):
        """根据每个维度的力大小自适应调整对应维度的K，力越大K越小

        Args:
            f: 死区处理后的力
        """
        K_adaptive = self.K_base.copy()

        # 死区处理后，力从0开始，所以直接用力的大小来计算
        force_max = self.threshold_high[0] * 4  # 力达到此值时K最小
        torque_max = self.threshold_high[1] * 4

        # 对平移的3个维度 (x, y, z) 分别计算
        for i in range(3):
            f_abs = np.abs(f[i])
            if f_abs > 0:
                ratio = f_abs / force_max
                ratio = np.clip(ratio, 0.0, 1.0)
                scale = 1.0 - ratio * (1.0 - self.K_min_ratio)
                K_adaptive[i, i] *= scale

        # 对旋转的3个维度 (rx, ry, rz) 分别计算
        for i in range(3, 6):
            t_abs = np.abs(f[i])
            if t_abs > 0:
                ratio = t_abs / torque_max
                ratio = np.clip(ratio, 0.0, 1.0)
                scale = 1.0 - ratio * (1.0 - self.K_min_ratio)
                K_adaptive[i, i] *= scale

        return K_adaptive

    def update(self, dt, tcp_pose, tcp_vel, target_pose, f_ext=None):

        if f_ext is None:
            f_ext = np.zeros(6)

        # 死区处理
        f = self._apply_deadzone(f_ext)

        # 自适应K：每个维度根据死区处理后的力单独调整
        self.K = self._compute_adaptive_K(f)

        # 计算当前TCP相对于目标位姿的偏移量 (在target坐标系下)
        # x = tcp_pose - target_pose (位姿误差)
        x_measured = calcPose6dError(target_pose, tcp_pose)

        # 使用测量值更新内部状态（滤波融合）
        alpha = 0.8  # 滤波系数，可调
        self.x = alpha * x_measured + (1 - alpha) * self.x
        self.x_dot = alpha * tcp_vel + (1 - alpha) * self.x_dot

        # 导纳动力学: M·ẍ + D·ẋ + K·x = f
        # 求解: ẍ = M⁻¹·(f - D·ẋ - K·x)
        x_ddot = np.linalg.inv(self.M) @ (f - self.D @ self.x_dot - self.K @ self.x)

        # 积分得到新的期望偏移
        self.x_dot += x_ddot * dt
        self.x = applyDeltaPose6d(self.x, self.x_dot * dt)

        # 将偏移量叠加到目标位姿上，得到新的目标位姿
        new_target_pose = applyDeltaPose6d(target_pose, self.x)

        return new_target_pose
