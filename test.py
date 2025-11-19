import numpy as np

from xlib.algo.controller import AdmittanceController
from xlib.algo.filter import MovingAverageFilter
from xlib.algo.utils.transforms import applyDeltaPose6d
from xlib.device.manipulator import UR
from xlib.device.sensor import Ft300sSensor

if __name__ == "__main__":
    ft_sensor = Ft300sSensor("/dev/ttyUSB0", timeout=1, zero_reset=True)
    ur_robot = UR(ip="172.16.11.233")
    M = np.diag([5.0, 5.0, 5.0, 0.05, 0.05, 0.05])
    D = np.diag([300.0, 300.0, 300.0, 8.0, 8.0, 8.0])
    K = np.diag([3000.0, 3000.0, 3000.0, 60.0, 60.0, 60.0])
    threshold_low = np.array([8.0, 0.3])
    threshold_high = np.array([10.0, 0.5])
    # threshold_high = np.array([100.0, 0.5])
    # threshold = np.array([5.0, 5.0, 5.0, 100, 100, 100])
    admittance_controller = AdmittanceController(M, D, K, threshold_high, threshold_low)
    desired_tcp_pose = ur_robot.tcp_pose
    filter = MovingAverageFilter(window_size=30)
    while True:
        start_time = ur_robot.initPeriod()
        ft_value = ft_sensor.get_force_torque()
        print("Raw Force Torque:", ft_value)
        cur_tcp_pose = ur_robot.tcp_pose
        filter.update(ft_value)
        ft_value = filter.output
        delta_pose = admittance_controller.update(1.0 / ur_robot.frequency, None, ft_value)
        new_pose = applyDeltaPose6d(desired_tcp_pose, delta_pose)
        ur_robot.servoTcp(new_pose, dt=1.0 / ur_robot.frequency)
        print("Force Torque:", ft_value)
        # print("TCP Velocity:", tcp_vel)
        print("Current TCP Pose:", cur_tcp_pose)
        print("New TCP Pose:", delta_pose)
        ur_robot.waitPeriod(start_time=start_time)
