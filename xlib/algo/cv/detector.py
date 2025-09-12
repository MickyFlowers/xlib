import numpy as np
from typing import Union
import cv2
import cv2.aruco as aruco
from ..utils.transforms import vecToMatrix
import logging


def get_single_aruco_pose(config: dict, img: np.ndarray, intrinsics_matrix, distortion):
    assert "aruco_type" in config
    assert "marker_size" in config
    aruco_dict = aruco.getPredefinedDictionary(config["aruco_type"])
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    try:
        # Detect ArUco markers
        corners, ids, rejected_markers = detector.detectMarkers(img)

        # Create image copy for visualization
        img_copy = img.copy()
        img_markers = aruco.drawDetectedMarkers(img_copy, corners, ids)

        # Check if exactly one marker was detected
        if ids is not None and len(ids) == 1:
            marker_size = config["marker_size"]
            # Define 3D coordinates of marker corners in marker-local coordinate system
            objp = np.array(
                [
                    [0, 0, 0],
                    [marker_size, 0, 0],
                    [marker_size, marker_size, 0],
                    [0, marker_size, 0],
                ],
                dtype=np.float32,
            )

            # Get 2D coordinates of detected marker corners
            imgp = corners[0][0]  # corners is shape (1, 1, 4, 2) for single marker

            # Solve PnP to get pose
            _, rvec, tvec = cv2.solvePnP(objp, imgp, intrinsics_matrix, distortion)

            # Draw 3D axes on the marker
            img_axes = cv2.drawFrameAxes(
                img_markers,
                intrinsics_matrix,
                distortion,
                rvec,
                tvec,
                marker_size / 2,  # Axis length is half marker size
            )

            # Convert rotation vector and translation vector to matrix
            return img_axes, vecToMatrix(rvec.squeeze(), tvec.squeeze())
        else:
            # No marker or multiple markers detected
            return img_markers, None
    except Exception as e:
        logging.exception(e)
        raise e


def get_aruco_pose(
    config: Union[list, dict], img: np.ndarray, intrinsics_matrix, distortion
):
    if isinstance(config, dict):
        assert all(
            [
                k in config
                for k in [
                    "aruco_type",
                    "num_markers",
                    "marker_size",
                    "marker_seperation",
                ]
            ]
        )
        aruco_type = config["aruco_type"]
        num_markers = config["num_markers"][0] * config["num_markers"][1]
        marker_size = config["marker_size"]
        marker_seperation = config["marker_seperation"]
        aruco_dict = aruco.getPredefinedDictionary(aruco_type)
        aruco_params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, aruco_params)
        board = aruco.GridBoard(
            config["num_markers"], marker_size, marker_seperation, aruco_dict
        )
        try:
            corners, ids, rejected_markers = detector.detectMarkers(img)
            corners, ids, rejected_markers, _ = detector.refineDetectedMarkers(
                img,
                board,
                corners,
                ids,
                rejected_markers,
                cameraMatrix=intrinsics_matrix,
                distCoeffs=distortion,
            )
            img_copy = img.copy()
            img_markers = aruco.drawDetectedMarkers(img_copy, corners, ids)
            if ids is not None and len(ids) == num_markers:
                objp, imgp = board.matchImagePoints(corners, ids)
                _, rvec, tvec = cv2.solvePnP(objp, imgp, intrinsics_matrix, distortion)
                img_axes = cv2.drawFrameAxes(
                    img_markers,
                    intrinsics_matrix,
                    distortion,
                    rvec,
                    tvec,
                    0.1,
                )
                return img_axes, vecToMatrix(rvec.squeeze(), tvec.squeeze())
            else:
                return img_markers, None
        except Exception as e:
            logging.exception(e)
            raise e
