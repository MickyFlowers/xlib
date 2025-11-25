import glob
import logging
import os

import cv2
import cv2.aruco as aruco
import numpy as np

from ..device.manipulator.manipulator_base import Manipulator
from ..device.sensor.camera import Camera
from .utils.transforms import *
from .utils.transforms import invPose6d


class EyeHandCalibrator:
    def __init__(self, camera: Camera, robot: Manipulator) -> None:
        self._camera = camera
        self._robot = robot
        self._aruco_dict = None
        self._aruco_params = None
        self._detector = None

    def setAruco(
        self,
        aruco_type=aruco.DICT_6X6_50,
        num_markers=(2, 2),
        marker_size=0.08,
        # marker_size=0.028,
        marker_seperation=0.01,
        # marker_seperation=0.004,
    ):
        logging.info("Setting aruco board")
        logging.info(f"aruco_type: {aruco_type}")
        logging.info(f"num_markers: {num_markers}")
        logging.info(f"marker_size: {marker_size}")
        logging.info(f"marker_seperation: {marker_seperation}")
        self.num_markers = num_markers[0] * num_markers[1]
        self._aruco_dict = aruco.getPredefinedDictionary(aruco_type)
        self._aruco_params = aruco.DetectorParameters()
        self._detector = aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        self._board = aruco.GridBoard(num_markers, marker_size, marker_seperation, self._aruco_dict)

    def calibrate(self, type, data_path):
        assert (
            self._aruco_dict is not None
            and self._aruco_params is not None
            and self._detector is not None
        )
        assert type in ["eye-to-hand", "eye-in-hand"]
        img_files = glob.glob(os.path.join(data_path, "*.jpg"))
        tcp_pose_files = glob.glob(os.path.join(data_path, "*.npy"))
        assert len(img_files) == len(tcp_pose_files)
        img_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
        tcp_pose_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
        tcp_poses = []
        aruco2camera_poses = []
        for img, pose in zip(img_files, tcp_pose_files):
            color_img = cv2.imread(img)
            tcp_pose = np.load(pose)
            _, aruco_pose = self._estimatePose(color_img)
            if type == "eye-to-hand":
                tcp_poses.append(np.linalg.inv(pose6dToMatrix(tcp_pose)))
            elif type == "eye-in-hand":
                tcp_poses.append(pose6dToMatrix(tcp_pose))
            aruco2camera_poses.append(aruco_pose)
        R_tcp2base = np.array([tcp_pose[:3, :3] for tcp_pose in tcp_poses])
        t_tcp2base = np.array([tcp_pose[:3, 3] for tcp_pose in tcp_poses])
        R_aruco2camera = np.array(
            [aruco2camera_pose[:3, :3] for aruco2camera_pose in aruco2camera_poses]
        )
        t_aruco2camera = np.array(
            [aruco2camera_pose[:3, 3] for aruco2camera_pose in aruco2camera_poses]
        )
        
        R_camera2tcp, t_camera2tcp = cv2.calibrateHandEye(
            R_tcp2base,
            t_tcp2base,
            R_aruco2camera,
            t_aruco2camera,
            method=cv2.CALIB_HAND_EYE_PARK,
        )

        camera2tcp_pose = np.eye(4)
        camera2tcp_pose[:3, :3] = R_camera2tcp
        camera2tcp_pose[:3, 3] = t_camera2tcp.squeeze()
        output_path = os.path.join(data_path, "result")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.save(os.path.join(output_path, "camera2tcp.npy"), camera2tcp_pose)
        logging.info("Calibrated camera to tcp transformation matrix")
        logging.info(f"Calibrated result\n{camera2tcp_pose}")

    def _estimatePose(self, img):
        try:
            corners, ids, rejected_markers = self._detector.detectMarkers(img)
            corners, ids, rejected_markers, _ = self._detector.refineDetectedMarkers(
                img,
                self._board,
                corners,
                ids,
                rejected_markers,
                cameraMatrix=self._camera.intrinsics_matrix,
                distCoeffs=self._camera.distortion,
            )
            img_copy = img.copy()
            img_markers = aruco.drawDetectedMarkers(img_copy, corners, ids)
            if ids is not None and len(ids) == self.num_markers:
                objp, imgp = self._board.matchImagePoints(corners, ids)
                _, rvec, tvec = cv2.solvePnP(
                    objp, imgp, self._camera.intrinsics_matrix, self._camera.distortion
                )
                img_axes = cv2.drawFrameAxes(
                    img_markers,
                    self._camera.intrinsics_matrix,
                    self._camera.distortion,
                    rvec,
                    tvec,
                    0.1,
                )
                return img_axes, vecCv2ToMatrix(rvec.squeeze(), tvec.squeeze())
            else:
                return img_markers, None
        except Exception as e:
            logging.exception(e)
            raise e

    def sampleImages(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        count = 0
        while True:
            color_img, _ = self._camera.get_frame()
            tcp_pose = self._robot.tcp_pose
            img_markers, aruco_pose = self._estimatePose(color_img)
            cv2.imshow("camera color img", img_markers)
            key = cv2.waitKey(1)
            if key & 0xFF == ord("s"):
                if aruco_pose is not None:
                    cv2.imwrite(os.path.join(output_path, f"img-{count}.jpg"), color_img)
                    np.save(os.path.join(output_path, f"pose-{count}.npy"), tcp_pose)
                    logging.info(f"Saved image-{count}.jpg and pose-{count}.npy")
                    count += 1
                else:
                    logging.error("No Aruco Marker Detected")
            elif key == ord("q"):
                logging.info("Exit Sample images")
                break
