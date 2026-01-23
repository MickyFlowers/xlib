import numpy as np
import pyrealsense2 as rs
import cv2
import logging
from ...algo.cv.detector import get_aruco_pose

class Camera:
    def __init__(self, width=640, height=480):
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.width = width
        self.height = height
        self.distortion = None
        self._color_img = None
        self._depth_img = None

    def get_frame(self):
        raise NotImplementedError("get_frame method is not implemented")

    @classmethod
    def load_param_from_file(cls, file_path, width=640, height=480):
        param_dict = np.load(file_path, allow_pickle=True).item()
        cls.fx = param_dict["fx"]
        cls.fy = param_dict["fy"]
        cls.cx = param_dict["cx"]
        cls.cy = param_dict["cy"]
        cls.width = width
        cls.height = height
        cls.distortion = param_dict["distortion"]
        cls._color_img = None
        cls._depth_img = None
        return cls

    @classmethod
    def load_parm(cls, fx, fy, cx, cy, width=640, height=480, distortion=None):
        cls.fx = fx
        cls.fy = fy
        cls.cx = cx
        cls.cy = cy
        cls.width = width
        cls.height = height
        cls.distortion = distortion
        cls._color_img = None
        cls._depth_img = None

        return cls

    def set_param(
        self, fx, fy, cx, cy, width=640, height=480, distortion=None, *args, **kwargs
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.distortion = distortion

    def set_param_from_file(self, file_path):
        param_dict = np.load(file_path, allow_pickle=True).item()
        self.fx = param_dict["fx"]
        self.fy = param_dict["fy"]
        self.cx = param_dict["cx"]
        self.cy = param_dict["cy"]
        self.width = param_dict["width"]
        self.height = param_dict["height"]
        self.distortion = param_dict["distortion"]

    @property
    def intrinsics_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    @property
    def color_img(self):
        return self._color_img

    @color_img.setter
    def color_img(self, color_img):
        self._color_img = np.asanyarray(color_img)

    @property
    def depth_img(self):
        return self._depth_img

    @depth_img.setter
    def depth_img(self, depth_img):
        self._depth_img = np.asanyarray(depth_img)

    def backproject(self) -> np.ndarray:
        if self._depth_img is not None:
            depth_img = self._depth_img
        else:
            raise ValueError("No depth frame provided")
        try:

            h, w = depth_img.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            x = ((u - self.cx) * depth_img / self.fx).astype(np.float32)
            y = ((v - self.cy) * depth_img / self.fy).astype(np.float32)
            z = depth_img.astype(np.float32)
        except:
            raise ValueError("Do not set camera parameters ")

        return np.stack([x, y, z], axis=-1)

    def projet(self, points: np.ndarray) -> np.ndarray:
        try:
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            u = (x * self.fx / z + self.cx).astype(np.int)
            v = (y * self.fy / z + self.cy).astype(np.int)
        except:
            raise ValueError("Do not set camera parameters")
        return np.stack([u, v], axis=-1)

    def pixel_to_camera_frame(self, pixel: np.ndarray):
        try:
            u, v = pixel[:, 0], pixel[:, 1]
            x = (u - self.cx) / self.fx
            y = (v - self.cy) / self.fy
        except:
            raise ValueError("Do not set camera parameters")
        return np.stack([x, y], axis=-1)

    def get_aruco_pose(self, config):
        return get_aruco_pose(
            config, self._color_img, self.intrinsics_matrix, self.distortion
        )


class RealSenseCamera(Camera):
    def __init__(
        self,
        color_width=640,
        color_height=480,
        depth_width=640,
        depth_height=480,
        frame_rate=30,
        short_range=None,
        exposure_time=None,
        align_to=rs.stream.color,
        serial_number=None,
    ):
        super().__init__()

        self.frame_rate = frame_rate
        self.pipeline = rs.pipeline()

        config = rs.config()
        if serial_number is not None:
            config.enable_device(serial_number)

        config.enable_stream(
            rs.stream.color, color_width, color_height, rs.format.bgr8, frame_rate
        )
        config.enable_stream(
            rs.stream.depth, depth_width, depth_height, rs.format.z16, frame_rate
        )
        cfg = self.pipeline.start(config)

        profile = self.pipeline.get_active_profile()
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        color_sensor = device.first_color_sensor()
        if exposure_time is not None:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, exposure_time)
        else:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        
        # if short_range:
        #     depth_sensor.set_option(rs.option.visual_preset, 5)
        # else:
        #     depth_sensor.set_option(rs.option.visual_preset, 3)
        self.depth_scale = depth_sensor.get_depth_scale()
        profile = cfg.get_stream(rs.stream.color)
        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        print(intrinsics)
        self.set_param(
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            width=intrinsics.width,
            height=intrinsics.height,
            distortion=np.array(intrinsics.coeffs),
        )
        logging.info("Camera Started:")
        logging.info(
            "--------------------------------Camera Parameters-----------------------------------"
        )
        logging.info(f"fx: {self.fx}")
        logging.info(f"fy: {self.fy}")
        logging.info(f"cx: {self.cx}")
        logging.info(f"cy: {self.cy}")
        logging.info(f"width: {self.width}")
        logging.info(f"height: {self.height}")
        logging.info(f"distortion: {self.distortion}")
        logging.info(
            "-------------------------------------------------------------------------------------"
        )
        self.align_to_color = rs.align(align_to)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align_to_color.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        depth_img = depth_img.astype(np.float32) * self.depth_scale
        self.color_img = color_img
        self.depth_img = depth_img
        return self.color_img, self.depth_img

    def close(self):
        self.pipeline.stop()

    def show(self):
        if self.color_img is not None:
            cv2.imshow("Color Image", self.color_img)
        if self.depth_img is not None:
            cv2.imshow("Depth Image", self.depth_img)

    def recordVideo(self, save_path):
        if not save_path.endswith(".avi"):
            raise ValueError("Only .avi format is supported")
        logging.info("Begin recording Video, Press Q to stop")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            save_path, fourcc, self.frame_rate, (self.width, self.height)
        )
        while True:
            self.get_frame()
            out.write(self.color_img)
            cv2.imshow("Color Image", self.color_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        out.release()
        logging.info("End recording Video")
        cv2.destroyAllWindows()

