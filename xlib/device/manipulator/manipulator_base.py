from abc import ABC, abstractmethod
import numpy as np


class Manipulator(ABC):
    @property
    @abstractmethod
    def tcp_pose(self) -> np.ndarray:
        """TCP位姿 (world坐标系下)"""
        pass

    @property
    @abstractmethod
    def tcp_velocity(self) -> np.ndarray:
        """TCP速度 (world坐标系下)"""
        pass

    @property
    @abstractmethod
    def joint_position(self) -> np.ndarray:
        """关节角度"""
        pass

    @abstractmethod
    def applyTcpVel(self, tcp_vel, acc, time):
        """施加TCP坐标系下的速度"""
        pass

    @abstractmethod
    def applyVel(self, vel, acc, time):
        """施加world坐标系下的速度"""
        pass

    @abstractmethod
    def moveToPose(self, pose, vel, acc, asynchronous):
        """移动到目标位姿 (world坐标系下)"""
        pass

    @abstractmethod
    def servoTcp(self, pose, dt):
        """伺服到目标位姿 (world坐标系下)"""
        pass
