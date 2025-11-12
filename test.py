import numpy as np

from xlib.algo.controller import AdmittanceController
from xlib.algo.filter import MovingAverageFilter
from xlib.algo.utils.transforms import velTransform
from xlib.device.manipulator import UR
from xlib.device.sensor import Ft300sSensor

if __name__ == "__main__":
    ft_sensor = Ft300sSensor("/dev/ttyUSB0", timeout=1, zero_reset=True)
    ur_robot = UR(ip="172.16.11.233")
    M = np.diag([0.3, 0.3, 0.3, 0.005, 0.005, 0.005])
    D = np.diag([15.0, 15.0, 15.0, 0.2, 0.2, 0.2])
    K = np.diag([300.0, 300.0, 300.0, 5.0, 5.0, 5.0])
    threshold_low = np.array([6.0, 0.3])
    threshold_high = np.array([8.0, 0.5])
    # threshold = np.array([5.0, 5.0, 5.0, 100, 100, 100])
    admittance_controller = AdmittanceController(M, D, K, threshold_high, threshold_low)
    desired_tcp_pose = ur_robot.tcp_pose_vec
    filter = MovingAverageFilter(window_size=30)
    while True:
        start_time = ur_robot.initPeriod()
        ft_value = ft_sensor.get_force_torque()
        # print("Raw Force Torque:", ft_value)
        tcp_vel = ur_robot.tcp_velocity
        cur_tcp_pose = ur_robot.tcp_pose_vec
        ft_value = velTransform(ft_value, ur_robot.tcp_pose[:3, :3])
        filter.update(ft_value)
        ft_value = filter.data
        new_tcp_pose, x_dot = admittance_controller.update(
            cur_tcp_pose, desired_tcp_pose, 1.0 / ur_robot.frequency, None, ft_value
        )
        ur_robot.servoTcp(new_tcp_pose, dt=1.0 / ur_robot.frequency)
        # print("Force Torque:", ft_value)
        # print("TCP Velocity:", tcp_vel)
        # print("Current TCP Pose:", cur_tcp_pose)
        # print("New TCP Pose:", new_tcp_pose)
        ur_robot.waitPeriod(start_time=start_time)
