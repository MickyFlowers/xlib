import numpy as np

from ..utils.transforms import applyDeltaPose6d, calcPose6dError


class AdmittanceController(object):
    def __init__(self, M, D, K, threshold_high, threshold_low, K_min_ratio=0.2, K_filter_alpha=0.05):
        """
        Args:
            M, D, K: 导纳参数
            threshold_high: 力激活阈值 [force, torque]
            threshold_low: 力死区阈值 [force, torque]
            K_min_ratio: 接触时K的最小比例 (0~1)，用于自适应刚度
            K_filter_alpha: K值滤波系数 (0~1)，越小越平滑，越大响应越快
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
        self.K_filter_alpha = K_filter_alpha
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
        force_max = self.threshold_high[0] * 3  # 力达到此值时K最小
        torque_max = self.threshold_high[1] * 3

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

        # 自适应K：每个维度根据死区处理后的力单独调整，并进行低通滤波平滑
        K_target = self._compute_adaptive_K(f)
        self.K = self.K_filter_alpha * K_target + (1 - self.K_filter_alpha) * self.K

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


class VelocityAdmittanceController(object):
    """基于速度输出的导纳控制器

    输入目标速度，输出经过导纳修正后的速度。
    适用于速度控制接口的机器人。
    """

    def __init__(self, M, D, threshold_high, threshold_low, vel_filter_alpha=0.8):
        """
        Args:
            M: 虚拟质量 [6,] 或 [6, 6]
            D: 虚拟阻尼 [6,] 或 [6, 6]
            threshold_high: 力激活阈值 [force, torque]
            threshold_low: 力死区阈值 [force, torque]
            vel_filter_alpha: 速度反馈滤波系数 (0~1)，越大越信任测量值
        """
        if not isinstance(M, np.ndarray):
            M = np.array(M)
            D = np.array(D)
            threshold_high = np.array(threshold_high)
            threshold_low = np.array(threshold_low)

        if M.ndim == 1:
            M = np.diag(M)
            D = np.diag(D)
        self.M = M
        self.D = D
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.vel_filter_alpha = vel_filter_alpha
        self.force_flag = False
        self.torque_flag = False
        self.x_dot = np.zeros(6)  # 导纳产生的速度偏移

    def reset(self):
        self.force_flag = False
        self.torque_flag = False
        self.x_dot = np.zeros(6)

    def update_params(self, M=None, D=None):
        if M is not None:
            self.M = M
        if D is not None:
            self.D = D

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

    def update(self, dt, target_vel, tcp_vel, f_ext=None):
        """
        基于速度的导纳控制更新（带速度反馈）

        Args:
            dt: 时间步长
            target_vel: 目标速度 [vx, vy, vz, wx, wy, wz]（来自策略网络或遥操作）
            tcp_vel: 当前TCP速度 [vx, vy, vz, wx, wy, wz]（本体反馈）
            f_ext: 外力 [fx, fy, fz, tx, ty, tz]

        Returns:
            output_vel: 导纳修正后的输出速度

        Note:
            target_vel, tcp_vel, f_ext 必须在同一坐标系下
        """
        if f_ext is None:
            f_ext = np.zeros(6)

        # 死区处理
        f = self._apply_deadzone(f_ext)

        # 计算当前速度偏移的测量值：实际速度 - 目标速度
        x_dot_measured = self.x_dot - target_vel

        # 滤波融合：测量值与积分值
        # self.x_dot = self.vel_filter_alpha * x_dot_measured + (1 - self.vel_filter_alpha) * self.x_dot

        # 一阶导纳动力学（无K项，纯速度控制）: M·ẍ + D·ẋ = f
        # ẍ = M⁻¹·(f - D·ẋ)
        x_ddot = np.linalg.inv(self.M) @ (f - self.D @ x_dot_measured)

        # 积分得到速度偏移
        self.x_dot += x_ddot * dt
        return self.x_dot
        # 输出速度 = 目标速度 + 导纳产生的速度偏移
        # output_vel = tcp_vel + x_ddot * dt

        # return output_vel

    def update_no_feedback(self, dt, target_vel, f_ext=None):
        """
        无速度反馈的导纳控制（纯积分模式）

        Args:
            dt: 时间步长
            target_vel: 目标速度 [vx, vy, vz, wx, wy, wz]
            f_ext: 外力 [fx, fy, fz, tx, ty, tz]

        Returns:
            output_vel: 导纳修正后的输出速度
        """
        if f_ext is None:
            f_ext = np.zeros(6)

        # 死区处理
        f = self._apply_deadzone(f_ext)

        # 一阶导纳动力学: M·ẍ + D·ẋ = f
        x_ddot = np.linalg.inv(self.M) @ (f - self.D @ self.x_dot)

        # 积分得到速度偏移
        self.x_dot += x_ddot * dt

        # 输出速度 = 目标速度 + 导纳产生的速度偏移
        output_vel = target_vel + self.x_dot

        return output_vel

    def update_no_target(self, dt, tcp_vel, f_ext=None):
        """
        纯导纳模式（无目标速度输入，仅根据力产生速度）

        Args:
            dt: 时间步长
            tcp_vel: 当前TCP速度（本体反馈）
            f_ext: 外力 [fx, fy, fz, tx, ty, tz]

        Returns:
            output_vel: 导纳产生的速度
        """
        return self.update(dt, np.zeros(6), tcp_vel, f_ext)
