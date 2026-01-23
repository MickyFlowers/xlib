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
        self._aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self._aruco_params.cornerRefinementWinSize = 5
        self._aruco_params.cornerRefinementMaxIterations = 30
        self._aruco_params.cornerRefinementMinAccuracy = 0.01
        self._detector = aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        self._board = aruco.GridBoard(num_markers, marker_size, marker_seperation, self._aruco_dict)

    def calibrate(
        self,
        type,
        data_path,
        method="auto",
        reproj_threshold=2.0,
        outlier_sigma=2.5,
    ):
        assert (
            self._aruco_dict is not None
            and self._aruco_params is not None
            and self._detector is not None
        )
        assert type in ["eye-to-hand", "eye-in-hand"]
        img_files = glob.glob(os.path.join(data_path, "*.jpg"))
        tcp_pose_files = glob.glob(os.path.join(data_path, "*.npy"))
        assert len(img_files) == len(tcp_pose_files)
        total_samples = len(img_files)
        img_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
        tcp_pose_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
        tcp_poses = []
        aruco2camera_poses = []
        for img, pose in zip(img_files, tcp_pose_files):
            color_img = cv2.imread(img)
            tcp_pose = np.load(pose)
            _, aruco_pose, reproj_error = self._estimatePose(color_img, return_error=True)
            if aruco_pose is None:
                logging.warning(f"Skip {os.path.basename(img)}: marker detection failed")
                continue
            if reproj_error is not None and reproj_error > reproj_threshold:
                logging.warning(
                    f"Skip {os.path.basename(img)}: reproj error {reproj_error:.3f}px"
                )
                continue
            if type == "eye-to-hand":
                tcp_poses.append(np.linalg.inv(pose6dToMatrix(tcp_pose)))
            elif type == "eye-in-hand":
                tcp_poses.append(pose6dToMatrix(tcp_pose))
            aruco2camera_poses.append(aruco_pose)
        if len(tcp_poses) < 3:
            raise ValueError("Not enough valid samples for hand-eye calibration")
        R_tcp2base = np.array([tcp_pose[:3, :3] for tcp_pose in tcp_poses])
        t_tcp2base = np.array([tcp_pose[:3, 3] for tcp_pose in tcp_poses])
        R_aruco2camera = np.array(
            [aruco2camera_pose[:3, :3] for aruco2camera_pose in aruco2camera_poses]
        )
        t_aruco2camera = np.array(
            [aruco2camera_pose[:3, 3] for aruco2camera_pose in aruco2camera_poses]
        )

        method_map = {
            "tsai": cv2.CALIB_HAND_EYE_TSAI,
            "park": cv2.CALIB_HAND_EYE_PARK,
            "horaud": cv2.CALIB_HAND_EYE_HORAUD,
            "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
            "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        if method == "auto":
            method_candidates = list(method_map.items())
        elif isinstance(method, str):
            if method not in method_map:
                raise ValueError(f"Unknown hand-eye method: {method}")
            method_candidates = [(method, method_map[method])]
        else:
            method_candidates = [("custom", method)]

        best_result = None
        rot_weight = 0.1
        for method_name, method_id in method_candidates:
            R_camera2tcp, t_camera2tcp = cv2.calibrateHandEye(
                R_tcp2base,
                t_tcp2base,
                R_aruco2camera,
                t_aruco2camera,
                method=method_id,
            )
            camera2tcp_pose = np.eye(4)
            camera2tcp_pose[:3, :3] = R_camera2tcp
            camera2tcp_pose[:3, 3] = t_camera2tcp.squeeze()
            trans_err, rot_err = self._hand_eye_errors(
                tcp_poses, aruco2camera_poses, camera2tcp_pose
            )
            score = float(np.median(trans_err) + rot_weight * np.median(rot_err))
            if best_result is None or score < best_result["score"]:
                best_result = {
                    "method_name": method_name,
                    "method_id": method_id,
                    "camera2tcp_pose": camera2tcp_pose,
                    "trans_err": trans_err,
                    "rot_err": rot_err,
                    "score": score,
                }
        if best_result is None:
            raise RuntimeError("Failed to compute hand-eye calibration")

        if outlier_sigma is not None and len(tcp_poses) >= 6:
            trans_err = best_result["trans_err"]
            rot_err = best_result["rot_err"]
            trans_thr = self._robust_threshold(trans_err, outlier_sigma)
            rot_thr = self._robust_threshold(rot_err, outlier_sigma)
            if trans_thr is not None and rot_thr is not None:
                keep = (trans_err <= trans_thr) & (rot_err <= rot_thr)
                if keep.sum() >= 3 and keep.sum() < len(keep):
                    logging.info(
                        f"Outlier rejection: keep {int(keep.sum())}/{len(keep)} samples"
                    )
                    tcp_poses = [pose for pose, ok in zip(tcp_poses, keep) if ok]
                    aruco2camera_poses = [
                        pose for pose, ok in zip(aruco2camera_poses, keep) if ok
                    ]
                    R_tcp2base = np.array([tcp_pose[:3, :3] for tcp_pose in tcp_poses])
                    t_tcp2base = np.array([tcp_pose[:3, 3] for tcp_pose in tcp_poses])
                    R_aruco2camera = np.array(
                        [pose[:3, :3] for pose in aruco2camera_poses]
                    )
                    t_aruco2camera = np.array(
                        [pose[:3, 3] for pose in aruco2camera_poses]
                    )
                    R_camera2tcp, t_camera2tcp = cv2.calibrateHandEye(
                        R_tcp2base,
                        t_tcp2base,
                        R_aruco2camera,
                        t_aruco2camera,
                        method=best_result["method_id"],
                    )
                    best_result["camera2tcp_pose"] = np.eye(4)
                    best_result["camera2tcp_pose"][:3, :3] = R_camera2tcp
                    best_result["camera2tcp_pose"][:3, 3] = t_camera2tcp.squeeze()

        camera2tcp_pose = best_result["camera2tcp_pose"]
        trans_err, rot_err = self._hand_eye_errors(
            tcp_poses, aruco2camera_poses, camera2tcp_pose
        )
        rot_err_deg = np.rad2deg(rot_err)
        output_path = os.path.join(data_path, "result")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.save(os.path.join(output_path, "camera2tcp.npy"), camera2tcp_pose)
        logging.info("Calibrated camera to tcp transformation matrix")
        logging.info(f"Hand-eye method: {best_result['method_name']}")
        logging.info(f"Samples used: {len(tcp_poses)}/{total_samples}")
        logging.info(
            "Translation residuals (m) mean/median/max: "
            f"{np.mean(trans_err):.6f}/{np.median(trans_err):.6f}/{np.max(trans_err):.6f}"
        )
        logging.info(
            "Rotation residuals (deg) mean/median/max: "
            f"{np.mean(rot_err_deg):.4f}/{np.median(rot_err_deg):.4f}/{np.max(rot_err_deg):.4f}"
        )
        logging.info(f"Calibrated result\n{camera2tcp_pose}")

    def _estimatePose(self, img, return_error=False):
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            corners, ids, rejected_markers = self._detector.detectMarkers(gray)
            corners, ids, rejected_markers, _ = self._detector.refineDetectedMarkers(
                gray,
                self._board,
                corners,
                ids,
                rejected_markers,
                cameraMatrix=self._camera.intrinsics_matrix,
                distCoeffs=self._camera.distortion,
            )
            img_copy = img.copy()
            img_markers = aruco.drawDetectedMarkers(img_copy, corners, ids)
            if ids is not None and len(ids) >= 4:
                objp, imgp = self._board.matchImagePoints(corners, ids)
                objp = np.asarray(objp, dtype=np.float64).reshape(-1, 3)
                imgp = np.asarray(imgp, dtype=np.float64).reshape(-1, 2)
                if len(objp) < 4:
                    if return_error:
                        return img_markers, None, None
                    return img_markers, None
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objp,
                    imgp,
                    self._camera.intrinsics_matrix,
                    self._camera.distortion,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    reprojectionError=2.0,
                    iterationsCount=100,
                )
                if not success:
                    if return_error:
                        return img_markers, None, None
                    return img_markers, None
                if inliers is not None and len(inliers) >= 4:
                    objp_in = objp[inliers[:, 0]]
                    imgp_in = imgp[inliers[:, 0]]
                else:
                    objp_in = objp
                    imgp_in = imgp
                if hasattr(cv2, "solvePnPRefineLM"):
                    rvec, tvec = cv2.solvePnPRefineLM(
                        objp_in,
                        imgp_in,
                        self._camera.intrinsics_matrix,
                        self._camera.distortion,
                        rvec,
                        tvec,
                    )
                elif hasattr(cv2, "solvePnPRefineVVS"):
                    rvec, tvec = cv2.solvePnPRefineVVS(
                        objp_in,
                        imgp_in,
                        self._camera.intrinsics_matrix,
                        self._camera.distortion,
                        rvec,
                        tvec,
                    )
                img_axes = cv2.drawFrameAxes(
                    img_markers,
                    self._camera.intrinsics_matrix,
                    self._camera.distortion,
                    rvec,
                    tvec,
                    0.1,
                )
                if return_error:
                    reproj_error = self._reprojection_error(
                        objp_in, imgp_in, rvec, tvec
                    )
                    return (
                        img_axes,
                        vecCv2ToMatrix(rvec.squeeze(), tvec.squeeze()),
                        reproj_error,
                    )
                return img_axes, vecCv2ToMatrix(rvec.squeeze(), tvec.squeeze())
            else:
                if return_error:
                    return img_markers, None, None
                return img_markers, None
        except Exception as e:
            logging.exception(e)
            raise e

    def _reprojection_error(self, objp, imgp, rvec, tvec):
        proj, _ = cv2.projectPoints(
            objp, rvec, tvec, self._camera.intrinsics_matrix, self._camera.distortion
        )
        proj = proj.reshape(-1, 2)
        imgp = imgp.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(proj - imgp, axis=1)))

    def _hand_eye_errors(self, tcp_poses, aruco2camera_poses, camera2tcp_pose):
        trans_err = []
        rot_err = []
        for tcp_pose, aruco_pose in zip(tcp_poses, aruco2camera_poses):
            left = tcp_pose @ camera2tcp_pose
            right = camera2tcp_pose @ aruco_pose
            delta = np.linalg.inv(left) @ right
            rot_vec, _ = cv2.Rodrigues(delta[:3, :3])
            rot_err.append(np.linalg.norm(rot_vec))
            trans_err.append(np.linalg.norm(delta[:3, 3]))
        return np.array(trans_err), np.array(rot_err)

    def _robust_threshold(self, values, sigma):
        if len(values) < 5:
            return None
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad <= 1e-12:
            return float(median)
        return float(median + sigma * 1.4826 * mad)

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
