import time

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R

from ...algo.utils.transforms import *
from .manipulator_base import Manipulator


class UR(Manipulator):
    def __init__(self, ip: str, base_to_world: np.ndarray = np.eye(4)) -> None:
        self.base_to_world = base_to_world
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        self.frequency = 125  # Hz

    @property
    def tcp_velocity(self) -> np.ndarray:
        vel_vec = self.rtde_r.getActualTCPSpeed()
        return np.array(vel_vec)

    @property
    def tcp_pose(self) -> np.ndarray:
        pose_vec = self.rtde_r.getActualTCPPose()
        trans_matrix = np.eye(4)
        rot_matrix = R.from_rotvec(pose_vec[3:]).as_matrix()
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[:3, 3] = pose_vec[:3]
        return trans_matrix

    @property
    def tcp_pose_vec(self) -> np.ndarray:
        pose_vec = self.rtde_r.getActualTCPPose()
        return np.array(pose_vec)

    @property
    def joint_position(self) -> np.ndarray:
        return self.rtde_r.getActualQ()

    @property
    def world_pose(self) -> np.ndarray:
        pose_vec = self.rtde_r.getActualTCPPose()
        trans_matrix = np.eye(4)
        rot_matrix = R.from_rotvec(pose_vec[3:]).as_matrix()
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[:3, 3] = pose_vec[:3]
        return self.base_to_world @ trans_matrix

    def servoTcp(self, pose, dt, lookahead_time=0.1, gain=300.0) -> None:
        if pose.shape == (4, 4):
            pos_vec = pose[:3, 3]
            rot_matrix = pose[:3, :3]
            rot_vec = R.from_matrix(rot_matrix).as_rotvec()
            pose = np.concatenate((pos_vec, rot_vec))
        self.rtde_c.servoL(pose, 0.5, 0.5, dt, lookahead_time, gain)

    def servoJoint(self, q, dt, vel=0.5, acc=0.5, lookahead_time=0.1, gain=300.0) -> None:
        self.rtde_c.servoJ(q, vel, acc, dt, lookahead_time, gain)

    def moveJoint(self, q, vel=1.0, acc=1.0, asynchronous=False):
        self.rtde_c.moveJ(q, vel, acc, asynchronous)

    def applyTcpVel(self, tcp_vel: np.ndarray, acc=1.0, time=0.0) -> None:
        pose_vec = self.rtde_r.getActualTCPPose()
        rot_matrix = R.from_rotvec(pose_vec[3:]).as_matrix()
        base_vel = velTransform(tcp_vel, rot_matrix)
        self.rtde_c.speedL(base_vel, acc, time)

    def applyVel(self, vel: np.ndarray, acc=1.0, time=0.0) -> None:
        self.rtde_c.speedL(vel, acc, time)

    def applyWorldVel(self, world_vel: np.ndarray, acc=1.0, time=0.0) -> None:
        pose_vec = self.rtde_r.getActualTCPPose()
        rot_matrix = np.linalg.inv(self.base_to_world[:3, :3])
        world_vel = velTransform(world_vel, rot_matrix)
        self.rtde_c.speedL(world_vel, acc, time)

    def stop(self, acc=10.0) -> None:
        self.rtde_c.stopL(acc)
        self.rtde_c.stopJ(acc)
        self.rtde_c.speedStop(acc)

    def disconnect(self) -> None:
        self.stop()
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    def moveToPose(self, pose, vel=0.25, acc=1.2, asynchronous=False):
        rot_matrix = pose[:3, :3]
        pos_vec = pose[:3, 3]
        rot_vec = R.from_matrix(rot_matrix).as_rotvec()
        pose = np.concatenate((pos_vec, rot_vec))
        self.rtde_c.moveL(pose, vel, acc, asynchronous)

    def moveToWorldPose(self, pose, vel=0.25, acc=1.2, asynchronous=False):
        world_pose = np.linalg.inv(self.base_to_world) @ pose
        self.moveToPose(world_pose, vel, acc, asynchronous)

    def moveToWorldErrorPose(self, pose, jnt_error, vel=0.25, acc=1.2, asynchronous=False):
        tcp_pose = np.linalg.inv(self.base_to_world) @ pose
        # self.moveToPose(tcp_pose, vel, acc, False)
        # time.sleep(0.2)
        # q = self.rtde_r.getActualQ()
        # q += jnt_error
        # self.rtde_c.moveJ(q, vel, acc, asynchronous)
        pos_vec = tcp_pose[:3, 3]
        rot_matrix = tcp_pose[:3, :3]
        rot_vec = R.from_matrix(rot_matrix).as_rotvec()
        pose_vec = np.concatenate((pos_vec, rot_vec))
        if not self.rtde_c.getInverseKinematicsHasSolution(pose_vec):
            return False
        q = self.rtde_c.getInverseKinematics(pose_vec)
        q_cur = self.rtde_r.getActualQ()
        if np.abs(np.array(q[:3]) - np.array(q_cur[:3])).max() > np.pi / 3 or q[2] > 0:
            return False
        else:
            q += jnt_error
            self.rtde_c.moveJ(q, vel, acc, asynchronous)
            return True

    def initPeriod(self):
        start_time = self.rtde_c.initPeriod()
        return start_time

    def waitPeriod(self, start_time):
        self.rtde_c.waitPeriod(start_time)
