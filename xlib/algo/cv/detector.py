import logging
from typing import Union

import cv2
import cv2.aruco as aruco
import numpy as np

from ..utils.transforms import vecCv2ToMatrix


def generate_single_aruco(
    aruco_type: int,
    marker_id: int,
    pixels: int,
    output_file: str,
):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type)
    marker_img = aruco.generateImageMarker(aruco_dict, marker_id, pixels)
    cv2.imwrite(output_file, marker_img)
    print(f"single aruco marker saved to: {output_file}")
    return True


def generate_aruco_board(
    rows=6,
    cols=6,
    total_markers=50,
    marker_size=15,
    marker_separation=2,
    output_file="aruco_board.png",
):

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)

    board = cv2.aruco.GridBoard(
        size=(cols, rows),
        markerLength=marker_size,
        markerSeparation=marker_separation,
        dictionary=aruco_dict,
        firstMarker=0,  # 从ID 0开始
    )

    pixels_per_mm = 10
    img_width = int((cols * marker_size + (cols - 1) * marker_separation) * pixels_per_mm)
    img_height = int((rows * marker_size + (rows - 1) * marker_separation) * pixels_per_mm)
    img = board.generateImage((img_width, img_height), marginSize=50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"6x6 ArUco Board (50 markers used) - Marker Size: {marker_size}mm"
    cv2.putText(img, text, (50, img_height - 20), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    print(f"aruco board saved to: {output_file}")
    print(f"尺寸: {img_width}x{img_height}像素")
    print(f"实际物理尺寸: {img_width/pixels_per_mm}x{img_height/pixels_per_mm}mm")

    # 显示图像
    cv2.imshow("ArUco Board", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_file, img)
    return img


def get_single_aruco_pose(config: dict, img: np.ndarray, intrinsics_matrix, distortion, rvec=None, tvec=None):
    assert "aruco_type" in config
    assert "marker_size" in config
    assert "id" in config
    aruco_dict = aruco.getPredefinedDictionary(config["aruco_type"])
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    try:
        # Detect ArUco markers
        corners, ids, rejected_markers = detector.detectMarkers(img)
        index = np.where(ids == config["id"])[0]
        corners = tuple([corners[i] for i in index])
        if index is None or len(index) != 1:
            return img, None
        ids = ids[index]
        # Create image copy for visualization
        img_copy = img.copy()
        img_markers = aruco.drawDetectedMarkers(img_copy, corners, ids)

        # Check if exactly one marker was detected
        if ids is not None and len(ids) == 1:
            marker_size = config["marker_size"]
            # Define 3D coordinates of marker corners in marker-local coordinate system
            objp = np.array(
                [
                    [-marker_size / 2, marker_size / 2, 0],
                    [marker_size / 2, marker_size / 2, 0],
                    [marker_size / 2, -marker_size / 2, 0],
                    [-marker_size / 2, -marker_size / 2, 0],
                ],
                dtype=np.float32,
            )

            # Get 2D coordinates of detected marker corners
            imgp = corners[0][0]  # corners is shape (1, 1, 4, 2) for single marker
            print(imgp)
            # Solve PnP to get pose
            if rvec is None or tvec is None:
                _, rvec, tvec = cv2.solvePnP(objp, imgp, intrinsics_matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
            else:
                _, rvec, tvec = cv2.solvePnP(
                    objp,
                    imgp,
                    intrinsics_matrix,
                    distortion,
                    useExtrinsicGuess=True,
                    rvec=rvec,
                    tvec=tvec,
                    flags=cv2.SOLVEPNP_IPPE,
                )

            # Draw 3D axes on the marker
            img_axes = cv2.drawFrameAxes(
                img_markers,
                intrinsics_matrix,
                distortion,
                rvec,
                tvec,
                marker_size,  # Axis length is half marker size
            )

            # Convert rotation vector and translation vector to matrix
            return img_axes, vecCv2ToMatrix(rvec.squeeze(), tvec.squeeze())
        else:
            # No marker or multiple markers detected
            return img_markers, None
    except Exception as e:
        logging.exception(e)
        raise e


def get_aruco_pose(config: Union[list, dict], img: np.ndarray, intrinsics_matrix, distortion):
    if isinstance(config, dict):
        assert all(
            [
                k in config
                for k in [
                    "aruco_type",
                    "num_markers",
                    "marker_size",
                    "marker_seperation",
                    "ids",
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
        board = aruco.GridBoard(config["num_markers"], marker_size, marker_seperation, aruco_dict)
        try:
            corners, ids, rejected_markers = detector.detectMarkers(img)
            index = np.where(ids == config["ids"])[0]
            if index is None or len(index) != num_markers:
                return img, None
            corners = tuple([corners[i] for i in index])
            ids = ids[index]
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
                return img_axes, vecCv2ToMatrix(rvec.squeeze(), tvec.squeeze())
            else:
                return img_markers, None
        except Exception as e:
            logging.exception(e)
            raise e


def get_single_aruco_pose_high_precision(
    config: dict, img: np.ndarray, intrinsics_matrix, distortion, rvec=None, tvec=None
):

    assert "aruco_type" in config, "Config must contain 'aruco_type'."
    assert "marker_size" in config, "Config must contain 'marker_size'."
    assert "id" in config, "Config must contain 'id'."

    aruco_dict = cv2.aruco.getPredefinedDictionary(config["aruco_type"])
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    try:
        corners, ids, rejected_markers = detector.detectMarkers(img)

        if ids is None:
            img_copy = img.copy()
            return img_copy, None

        index = np.where(ids == config["id"])[0]
        if len(index) != 1:
            img_copy = img.copy()
            img_markers = cv2.aruco.drawDetectedMarkers(img_copy, corners, ids)
            return img_markers, None

        marker_corners = corners[index[0]]  # shape (1, 4, 2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        corners_subpix = cv2.cornerSubPix(
            gray,
            marker_corners.astype(np.float32),
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
        marker_size = config["marker_size"]
        objp = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        imgp = corners_subpix[0]
        success, rvec, tvec = cv2.solvePnP(
            objp,
            imgp,
            intrinsics_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,  # 使用IPPE算法
        )

        if not success:
            logging.warning("IPPE failed. Trying with default method.")
            success, rvec, tvec = cv2.solvePnP(objp, imgp, intrinsics_matrix, distortion)

        if success:
            img_copy = img.copy()
            img_markers = cv2.aruco.drawDetectedMarkers(img_copy, [corners_subpix], np.array([[config["id"]]]))

            axis_length = marker_size / 2.0
            img_axes = cv2.drawFrameAxes(img_markers, intrinsics_matrix, distortion, rvec, tvec, axis_length)
            transform_matrix = vecCv2ToMatrix(rvec.squeeze(), tvec.squeeze())
            return img_axes, transform_matrix
        else:
            # PnP求解失败
            img_copy = img.copy()
            img_markers = cv2.aruco.drawDetectedMarkers(img_copy, [corners_subpix], np.array([[config["id"]]]))
            logging.warning("Failed to solve PnP for the detected marker.")
            return img_markers, None

    except Exception as e:
        logging.exception(f"An error occurred during pose estimation: {e}")
        # 在发生异常时，至少返回原始图像，防止程序崩溃
        return img.copy(), None
