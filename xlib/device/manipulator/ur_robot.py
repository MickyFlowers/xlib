import time

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R

from ...algo.utils.transforms import *
from .manipulator_base import Manipulator


class UR(Manipulator):
    def __init__(self, ip: str, base_to_world: np.ndarray = np.zeros(6)) -> None:
        self.base_to_world = base_to_world
        self.base_to_world_mtx = pose6dToMatrix(self.base_to_world)
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        self.frequency = 125  # 
        self._init()
    
    def _init(self):
        self.servo_target_tcp = self.tcp_pose
        self.servo_target_world_tcp = self.world_pose

    @property
    def tcp_velocity(self) -> np.ndarray:
        vel_vec = self.rtde_r.getActualTCPSpeed()
        return np.array(vel_vec)

    @property
    def tcp_pose(self) -> np.ndarray:
        pose_vec = self.rtde_r.getActualTCPPose()
        return np.array(pose_vec)

    @property
    def joint_position(self) -> np.ndarray:
        jnt_vec = self.rtde_r.getActualQ()
        return np.array(jnt_vec)

    @property
    def world_pose(self) -> np.ndarray:
        pose_vec = self.rtde_r.getActualTCPPose()
        trans_matrix = pose6dToMatrix(pose_vec)
        return matrixToPose6d(self.base_to_world_mtx @ trans_matrix)
    
    @property
    def target_pose(self) -> np.ndarray:
        pose_vec = self.rtde_r.getTargetTCPPose()
        return np.array(pose_vec)   
    
    @property
    def world_target_pose(self) -> np.ndarray:
        pose_vec = self.rtde_r.getTargetTCPPose()
        trans_matrix = pose6dToMatrix(pose_vec)
        return matrixToPose6d(self.base_to_world_mtx @ trans_matrix)
    
    def reset_servo_target_tcp(self) -> None:
        self.servo_target_tcp = self.tcp_pose
        self.servo_target_world_tcp = self.world_pose

        
    def servoTcp(self, pose, dt, lookahead_time=0.1, gain=800.0, max_pos_vel=0.1, max_rot_vel=0.5) -> None:
        assert pose.shape == (6,)
        delta_pose = calcPose6dError(self.servo_target_tcp, pose)
        vel = delta_pose / dt
        if np.linalg.norm(vel[:3]) > max_pos_vel:
            vel[:3] = vel[:3] / np.linalg.norm(vel[:3]) * max_pos_vel
        if np.linalg.norm(vel[3:]) > max_rot_vel:
            vel[3:] = vel[3:] / np.linalg.norm(vel[3:]) * max_rot_vel
        self.servo_target_tcp = applyDeltaPose6d(self.servo_target_tcp, vel * dt)
        self.rtde_c.servoL(self.servo_target_tcp, 0.5, 0.5, dt, lookahead_time, gain)
    
    def servoWorldTcp(self, pose, dt, lookahead_time=0.1, gain=800.0, max_pos_vel=0.1, max_rot_vel=0.5):
        assert pose.shape == (6,)
        delta_pose = calcPose6dError(self.servo_target_world_tcp, pose)
        vel = delta_pose / dt
        if np.linalg.norm(vel[:3]) > max_pos_vel:
            vel[:3] = vel[:3] / np.linalg.norm(vel[:3]) * max_pos_vel
        if np.linalg.norm(vel[3:]) > max_rot_vel:
            vel[3:] = vel[3:] / np.linalg.norm(vel[3:]) * max_rot_vel
        self.servo_target_world_tcp = applyDeltaPose6d(self.servo_target_world_tcp, vel * dt)
        world_pose = np.linalg.inv(self.base_to_world_mtx) @ pose6dToMatrix(self.servo_target_world_tcp)
        world_pose = matrixToPose6d(world_pose)
        self.rtde_c.servoL(world_pose, 0.5, 0.5, dt, lookahead_time, gain)
    
    
    def servoJoint(self, q, dt, vel=0.5, acc=0.5, lookahead_time=0.1, gain=300.0) -> None:
        self.rtde_c.servoJ(q, vel, acc, dt, lookahead_time, gain)

    def moveJoint(self, q, vel=1.0, acc=1.0, asynchronous=False):
        self.rtde_c.moveJ(q, vel, acc, asynchronous)

    def applyTcpVel(self, tcp_vel: np.ndarray, acc=1.0, time=0.0) -> None:
        pose_vec = self.rtde_r.getActualTCPPose()
        base_vel = velTransform(tcp_vel, pose_vec[3:])
        self.rtde_c.speedL(base_vel, acc, time)

    def applyVel(self, vel: np.ndarray, acc=1.0, time=0.0) -> None:
        self.rtde_c.speedL(vel, acc, time)

    def applyWorldVel(self, world_vel: np.ndarray, acc=1.0, time=0.0) -> None:
        rot_vec = invPose6d(self.base_to_world)[3:]
        world_vel = velTransform(world_vel, rot_vec)
        self.rtde_c.speedL(world_vel, acc, time)

    def stop(self, acc=10.0) -> None:
        self.rtde_c.stopL(acc)
        self.rtde_c.stopJ(acc)
        self.rtde_c.speedStop(acc)

    def servoStop(self, acc=10.0) -> None:
        self.rtde_c.servoStop(acc)
        
    def close(self) -> None:
        self.stop()
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    def moveToPose(self, pose, vel=0.25, acc=1.2, asynchronous=False):
        assert pose.shape == (6,)
        self.rtde_c.moveL(pose, vel, acc, asynchronous)

    def moveToWorldPose(self, pose, vel=0.25, acc=1.2, asynchronous=False):
        world_pose = np.linalg.inv(self.base_to_world_mtx) @ pose6dToMatrix(pose)
        world_pose = matrixToPose6d(world_pose)
        self.moveToPose(world_pose, vel, acc, asynchronous)

    def moveToWorldErrorPose(self, pose, jnt_error, vel=0.25, acc=1.2, asynchronous=False):
        assert pose.shape == (6,), "Input pose should be 6D pose vector"
        world_pose_matrix = pose6dToMatrix(pose)
        tcp_pose = np.linalg.inv(self.base_to_world_mtx) @ world_pose_matrix
        # self.moveToPose(tcp_pose, vel, acc, False)
        # time.sleep(0.2)
        # q = self.rtde_r.getActualQ()
        # q += jnt_error
        # self.rtde_c.moveJ(q, vel, acc, asynchronous)
        pose_vec = matrixToPose6d(tcp_pose)
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

