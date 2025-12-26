import time

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R

from ...algo.utils.transforms import *
from .manipulator_base import Manipulator


class UR(Manipulator):
    def __init__(self, ip: str, base_in_world: np.ndarray = np.zeros(6)) -> None:
        self._base_in_world = base_in_world
        self._base_in_world_mtx = pose6dToMatrix(self._base_in_world)
        self._world_in_base_mtx = np.linalg.inv(self._base_in_world_mtx)
        self._world_in_base = matrixToPose6d(self._world_in_base_mtx)
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        self.frequency = 125
        self._init()

    def _init(self):
        self._servo_target_pose = self.tcp_pose

    def _pose_base_to_world(self, pose_in_base: np.ndarray) -> np.ndarray:
        """将base坐标系下的位姿转换到world坐标系"""
        pose_mtx = pose6dToMatrix(pose_in_base)
        return matrixToPose6d(self._base_in_world_mtx @ pose_mtx)

    def _pose_world_to_base(self, pose_in_world: np.ndarray) -> np.ndarray:
        """将world坐标系下的位姿转换到base坐标系"""
        pose_mtx = pose6dToMatrix(pose_in_world)
        return matrixToPose6d(self._world_in_base_mtx @ pose_mtx)

    def _vel_base_to_world(self, vel_in_base: np.ndarray) -> np.ndarray:
        """将base坐标系下的速度转换到world坐标系"""
        return velTransform(vel_in_base, self._base_in_world[3:])

    def _vel_world_to_base(self, vel_in_world: np.ndarray) -> np.ndarray:
        """将world坐标系下的速度转换到base坐标系"""
        return velTransform(vel_in_world, self._world_in_base[3:])

    @property
    def tcp_pose(self) -> np.ndarray:
        """TCP位姿 (world坐标系下)"""
        pose_in_base = self.rtde_r.getActualTCPPose()
        return self._pose_base_to_world(np.array(pose_in_base))

    @property
    def tcp_velocity(self) -> np.ndarray:
        """TCP速度 (world坐标系下)"""
        vel_in_base = self.rtde_r.getActualTCPSpeed()
        return self._vel_base_to_world(np.array(vel_in_base))

    @property
    def tcp_velocity_in_tcp(self) -> np.ndarray:
        """TCP速度 (TCP坐标系下)"""
        vel_in_base = self.rtde_r.getActualTCPSpeed()
        pose_in_base = self.rtde_r.getActualTCPPose()
        tcp_in_base = pose_in_base[3:]
        base_in_tcp = R.from_rotvec(tcp_in_base).inv().as_rotvec()
        return velTransform(np.array(vel_in_base), base_in_tcp)

    @property
    def joint_position(self) -> np.ndarray:
        """关节角度"""
        return np.array(self.rtde_r.getActualQ())

    @property
    def target_pose(self) -> np.ndarray:
        """目标TCP位姿 (world坐标系下)"""
        pose_in_base = self.rtde_r.getTargetTCPPose()
        return self._pose_base_to_world(np.array(pose_in_base))

    def reset_servo_target(self) -> None:
        """重置servo目标为当前位姿"""
        self._servo_target_pose = self.tcp_pose

    def servoTcp(self, pose, dt, lookahead_time=0.1, gain=1500.0, max_pos_vel=0.1, max_rot_vel=0.5) -> None:
        """伺服到目标位姿 (world坐标系下)"""
        assert pose.shape == (6,)
        delta_pose = calcPose6dError(self._servo_target_pose, pose)
        vel = delta_pose / dt
        if np.linalg.norm(vel[:3]) > max_pos_vel:
            vel[:3] = vel[:3] / np.linalg.norm(vel[:3]) * max_pos_vel
        if np.linalg.norm(vel[3:]) > max_rot_vel:
            vel[3:] = vel[3:] / np.linalg.norm(vel[3:]) * max_rot_vel
        self._servo_target_pose = applyDeltaPose6d(self._servo_target_pose, vel * dt)
        pose_in_base = self._pose_world_to_base(self._servo_target_pose)
        self.rtde_c.servoL(pose_in_base, 0.5, 0.5, dt, lookahead_time, gain)

    def servoJoint(self, q, dt, vel=0.5, acc=0.5, lookahead_time=0.1, gain=300.0) -> None:
        """伺服到目标关节角度"""
        self.rtde_c.servoJ(q, vel, acc, dt, lookahead_time, gain)

    def moveJoint(self, q, vel=1.0, acc=1.0, asynchronous=False):
        """移动到目标关节角度"""
        self.rtde_c.moveJ(q, vel, acc, asynchronous)

    def applyTcpVel(self, vel_in_tcp: np.ndarray, acc=1.0, time=0.0) -> None:
        """施加TCP坐标系下的速度"""
        pose_in_base = self.rtde_r.getActualTCPPose()
        tcp_in_base = pose_in_base[3:]
        vel_in_base = velTransform(vel_in_tcp, tcp_in_base)
        self.rtde_c.speedL(vel_in_base, acc, time)

    def applyVel(self, vel_in_world: np.ndarray, acc=1.0, time=0.0) -> None:
        """施加world坐标系下的速度"""
        vel_in_base = self._vel_world_to_base(vel_in_world)
        self.rtde_c.speedL(vel_in_base, acc, time)

    def stop(self, acc=10.0) -> None:
        """停止所有运动"""
        self.rtde_c.stopL(acc)
        self.rtde_c.stopJ(acc)
        self.rtde_c.speedStop(acc)

    def servoStop(self, acc=10.0) -> None:
        """停止伺服运动"""
        self.rtde_c.servoStop(acc)

    def close(self) -> None:
        """关闭连接"""
        self.stop()
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    def moveToPose(self, pose_in_world, vel=0.25, acc=1.2, asynchronous=False):
        """移动到目标位姿 (world坐标系下)"""
        assert pose_in_world.shape == (6,)
        pose_in_base = self._pose_world_to_base(pose_in_world)
        self.rtde_c.moveL(pose_in_base, vel, acc, asynchronous)

    def initPeriod(self):
        """初始化周期计时"""
        return self.rtde_c.initPeriod()

    def waitPeriod(self, start_time):
        """等待周期结束"""
        self.rtde_c.waitPeriod(start_time)
