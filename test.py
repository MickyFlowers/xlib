from xlib.algo.cv.detector import get_single_aruco_pose
import cv2
import numpy as np
import cv2.aruco as aruco

if __name__ == "__main__":
    from xlib.device.sensor.camera import RealSenseCamera

    camera = RealSenseCamera()
    config = {
        "aruco_type": aruco.DICT_6X6_250,
        "num_markers": (1, 1),
        "marker_size": 0.015,
        "marker_seperation": 0.02,
    }
    while True:
        color_img, _ = camera.get_frame()
        img_axes, pose = get_single_aruco_pose(
            config, color_img, camera.intrinsics_matrix, camera.distortion
        )
        if pose is not None:
            print(pose)
        cv2.imshow("img", img_axes)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
