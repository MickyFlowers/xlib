from .. import kp_matcher
from ....device.sensor.camera import Camera
import numpy as np
from ...utils import metric
import cv2
import copy
from .vs_controller_base import VisualServoControllerBase
from .cnsv2_simple_cilent import SimpleClient


class CNSV2(VisualServoControllerBase):
    def __init__(self, camera: Camera, model="CNSv2_m0117", *args, **kwargs):
        self.camera = camera
        self.cur_img = None
        self.tar_img = None
        self.controller = SimpleClient(model)

    def update(self, *args, **kwargs):
        if "cur_img" in kwargs:
            assert isinstance(
                kwargs["cur_img"], np.ndarray
            ), "Image should be a numpy array"
            self.cur_img = kwargs["cur_img"]
        if "tar_img" in kwargs:
            assert isinstance(
                kwargs["tar_img"], np.ndarray
            ), "Image should be a numpy array"
            self.tar_img = kwargs["tar_img"]
            

    def calc_vel(self, depth_hint, mask=None):
        color_image = copy.deepcopy(self.cur_img)
        tar_image = copy.deepcopy(self.tar_img)
        score = metric.calc_ssim(tar_image, color_image)
        tar_image[mask] = 0
        self.controller.set_target(tar_image, self.camera.intrinsics_matrix, depth_hint=depth_hint, None)
        vel = self.controller.get_control_rate(color_image)
        
        match_img = np.concatenate([tar_image, color_image], axis=1)
        return True, vel, score, match_img


class IBVS(VisualServoControllerBase):
    def __init__(
        self, camera: Camera, kp_algo=kp_matcher.KpMatchAlgo, *args, **kwargs
    ) -> None:
        self.kp_algo = kp_algo(*args, **kwargs)
        self.camera = camera
        self.cur_img = None
        self.tar_img = None
        self.cur_depth = None
        self.tar_depth = None

    def update(self, *args, **kwargs):

        if "cur_img" in kwargs:
            assert isinstance(
                kwargs["cur_img"], np.ndarray
            ), "Image should be a numpy array"
            self.cur_img = kwargs["cur_img"]
        if "tar_img" in kwargs:
            assert isinstance(
                kwargs["tar_img"], np.ndarray
            ), "Image should be a numpy array"
            self.tar_img = kwargs["tar_img"]
        if "cur_depth" in kwargs:
            assert isinstance(
                kwargs["cur_depth"], np.ndarray
            ), "Depth should be a numpy array"
            self.cur_depth = kwargs["cur_depth"]
        if "tar_depth" in kwargs:
            assert isinstance(
                kwargs["tar_depth"], np.ndarray
            ), "Depth should be a numpy array"
            self.tar_depth = kwargs["tar_depth"]

    def cal_vel_from_kp(self, tar_kp, cur_kp, tar_z, cur_z):

        assert tar_kp.shape == cur_kp.shape, "Keypoints shape mismatch"
        tar_kp = self.camera.pixel_to_camera_frame(tar_kp)
        cur_kp = self.camera.pixel_to_camera_frame(cur_kp)
        num_kp = tar_kp.shape[0]

        cur_x = cur_kp[:, 0]
        cur_y = cur_kp[:, 1]

        cur_L = np.zeros((num_kp * 2, 6), cur_kp.dtype)
        cur_L[0::2, 0] = -1.0 / cur_z
        cur_L[0::2, 2] = cur_x / cur_z
        cur_L[0::2, 3] = cur_x * cur_y
        cur_L[0::2, 4] = -(1 + cur_x * cur_x)
        cur_L[0::2, 5] = cur_y
        cur_L[1::2, 1] = -1.0 / cur_z
        cur_L[1::2, 2] = cur_y / cur_z
        cur_L[1::2, 3] = 1 + cur_y * cur_y
        cur_L[1::2, 4] = -cur_x * cur_y
        cur_L[1::2, 5] = -cur_x

        tar_x = tar_kp[:, 0]
        tar_y = tar_kp[:, 1]

        tar_L = np.zeros((num_kp * 2, 6), tar_kp.dtype)
        tar_L[0::2, 0] = -1.0 / tar_z
        tar_L[0::2, 2] = tar_x / tar_z
        tar_L[0::2, 3] = tar_x * tar_y
        tar_L[0::2, 4] = -(1 + tar_x * tar_x)
        tar_L[0::2, 5] = tar_y
        tar_L[1::2, 1] = -1.0 / tar_z
        tar_L[1::2, 2] = tar_y / tar_z
        tar_L[1::2, 3] = 1 + tar_y * tar_y
        tar_L[1::2, 4] = -tar_x * tar_y
        tar_L[1::2, 5] = -tar_x

        error = np.zeros(num_kp * 2, cur_kp.dtype)
        error[0::2] = tar_x - cur_x
        error[1::2] = tar_y - cur_y

        mean_L = (cur_L + tar_L) / 2.0
        vel = np.linalg.lstsq(mean_L, error)[0]
        return vel

    def calc_vel(self, mask=None, use_median_depth=False):
        assert (
            self.kp_algo is not None and self.camera is not None
        ), "KeyPoint Extractor or Camera not provided"
        try:
            tar_kp, cur_kp, match_img = self.kp_algo.match(
                self.tar_img, self.cur_img, mask, True, self.camera
            )
            tar_kp_int = np.floor(tar_kp).astype(int)
            cur_kp_int = np.floor(cur_kp).astype(int)
            tar_kp_x = tar_kp_int[:, 0]
            tar_kp_y = tar_kp_int[:, 1]
            cur_kp_x = cur_kp_int[:, 0]
            cur_kp_y = cur_kp_int[:, 1]

            tar_depth = self.tar_depth.squeeze()
            cur_depth = self.cur_depth.squeeze()

            tar_z = tar_depth[tar_kp_y, tar_kp_x]
            cur_z = cur_depth[cur_kp_y, cur_kp_x]
            if use_median_depth:
                tar_z = np.median(tar_z)
                cur_z = np.median(cur_z)

            vel = self.cal_vel_from_kp(tar_kp, cur_kp, tar_z, cur_z)
            score = metric.calc_ssim(self.tar_img, self.cur_img)
            cv2.putText(
                match_img,
                "SSIM score: {:.3f}".format(score),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (100, 100, 100),
                2,
            )
            return True, vel, score, match_img
        except Exception as e:
            return False, None, None, None
