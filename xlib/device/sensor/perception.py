import numpy as np


class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    @property
    def K(self):
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]]
        )

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5]
        )
        return intrinsic
    
    @classmethod
    def default(cls):
        return cls(width=640, height=480, fx=540, fy=540, cx=320, cy=240)

    def pixel_to_norm_camera_plane(self, uv: np.ndarray):
        xy = (uv - np.array([self.cx, self.cy])) / np.array([self.fx, self.fy])
        return xy
    
    def norm_camera_plane_to_pixel(self, xy: np.ndarray, clip=True, round=False):
        uv = xy * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])
        if clip: uv = np.clip(uv, 0, [self.width-1, self.height-1])
        if round: uv = np.round(uv).astype(np.int32)
        return uv
    
    # def inv_proj(self, depth: np.ndarray):
    #     _, _, H, W = depth.shape
    #     XX, YY = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    #     z = depth
    #     x = (XX - self.cx) / self.fx * z
    #     y = (YY - self.cy) / self.fy * z
    #     pcd = np.concatenate([x, y, z], axis=1)
    #     return pcd  # (B, 1, H, W)

    def inv_project(self, uv: np.ndarray, depth: np.ndarray, wcT: np.ndarray = None):
        """
        - uv: (N, 2), pixel coordinates
        - depth: (N,)
        - wcT: (4, 4)
        """
        z = depth[..., None]
        xy = (uv - np.array([self.cx, self.cy])) / np.array([self.fx, self.fy]) * z
        pcd = np.concatenate([xy, z], axis=-1)

        if wcT is not None:
            pcd = pcd @ wcT[:3, :3].T + wcT[:3, 3]
        return pcd  # (N, 3)
    
    def project(self, xyz: np.ndarray, wcT: np.ndarray = None):
        """
        - xyz: (N, 3), 3D points
        - wcT: (4, 4)
        """
        if wcT is not None:
            cwT = np.linalg.inv(wcT)
            xyz = xyz @ cwT[:3, :3].T + cwT[:3, 3]
        uv = xyz[..., :2] / (xyz[..., 2:3] + 1e-16)
        uv = uv * np.array([self.fx, self.fy]) + np.array([self.cx, self.cy])
        return uv


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, intrinsic: CameraIntrinsic, near=0.01, far=4):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.gl_proj_matrix = self.proj_matrix.flatten(order="F")

    def to_dict(self):
        data = {
            "intrinsic": self.intrinsic.to_dict(),
            "near": self.near, 
            "far": self.far, 
        }
        return data

    @classmethod
    def from_dict(cls, data: dict):
        intrinsic = CameraIntrinsic.from_dict(data["intrinsic"])
        near = data["near"]
        far = data["far"]
        return cls(intrinsic, near, far)

    def to_isaac(self, pixel_size=3e-3):
        return dict(
            resolution = (self.intrinsic.width, self.intrinsic.height),
            horizontal_aperture = pixel_size * self.intrinsic.width,
            vertical_aperture = pixel_size * self.intrinsic.height,
            focal_length_x = self.intrinsic.fx * pixel_size,
            focal_length_y = self.intrinsic.fy * pixel_size,
            clipping_range = (self.near, self.far)
        )

    def render_bullet(self, extrinsic: np.ndarray, client=0):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref (^{cam}_{world} T).
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.copy() if extrinsic is not None else np.eye(4)
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")

        import pybullet as p
        result = p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=self.gl_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client
        )

        rgb = np.ascontiguousarray(result[2][:, :, :3])
        z_buffer = result[3]
        seg = result[4]

        depth = self.far * self.near / (
            self.far - (self.far - self.near) * z_buffer)

        return Frame(self, rgb, depth, seg, extrinsic)

    def project(self, extrinsic, points) -> np.ndarray:
        """Project 3d points to pixel coordinates

        Arguments:
        - extrinsic: (4, 4), cwT
        - points: (N, 3), points in world frame

        Returns:
        - uv: (N, 2), pixel coordinates.
            If points are calculated from the the same frame, then: 
                range of u: [0, W-1];
                range of v: [0, H-1];
        """
        W, H = self.intrinsic.width, self.intrinsic.height
        N = len(points)
        points_homo = np.concatenate([points, np.ones((N, 1))], axis=1)

        proj = self.proj_matrix @ extrinsic @ points_homo.T  # (4, N)
        proj = (proj / proj[-1, :]).T  # (N, 4)
        proj[:, 0] = -proj[:, 0]
        uv = (proj[:, :2] + 1.) * np.array([W, H]) / 2.
        return uv
    
    def inv_project(self, extrinsic, uv, Z) -> np.ndarray:
        """Inverse projection, get corresponding 3d positions of pixel coordinates

        Arguments:
        - extrinsic: (4, 4), ^{cam} _{world} T
        - uv: (N, 2), pixel coordinates,
                range of u: [0, W-1];
                range of v: [0, H-1];
        - Z: (N,), depth in camera frame
        
        Returns:
        - points: (N, 3)
        """
        W, H = self.intrinsic.width, self.intrinsic.height
        N = len(uv)
        inv_proj = uv * 2. / np.array([W, H]) - 1.
        inv_proj[:, 0] = -inv_proj[:, 0]

        f, n = self.far, self.near
        norm_Z = (f+n)/(f-n) + 2*n*f/(f-n) * 1./Z

        inv_proj = np.concatenate([inv_proj, norm_Z[:, None], np.ones((N, 1))], axis=-1)
        inv_proj = inv_proj * -Z[:, None]  # (N, 4)

        X = (inv_proj[:, 0] - Z*self.proj_matrix[0, 2]) / self.proj_matrix[0, 0]
        Y = (inv_proj[:, 1] - Z*self.proj_matrix[1, 2]) / self.proj_matrix[1, 1]
        points = np.stack([X, Y, Z, np.ones_like(Z)], axis=0)  # (4, N)
        points = np.linalg.inv(extrinsic) @ points
        points = np.ascontiguousarray(points[:3, :].T)

        # # equal implementation
        # inv_proj = np.linalg.inv(self.proj_matrix @ extrinsic) @ inv_proj.T  # (4, N)
        # points = inv_proj[:3, :].T
        return points


class Frame(object):
    def __init__(
        self, 
        camera: Camera, 
        rgb: np.ndarray, 
        depth: np.ndarray, 
        seg=None, 
        extrinsic=None
    ):
        self.camera = camera
        self.rgb = rgb
        self.depth = depth
        self.seg = seg
        self.extrinsic = np.eye(4) if extrinsic is None else extrinsic

        # cache
        self._pc_camera = None
        self._pc_world = None
    
    def _pointcloud(self, extrinsic):
        H, W = self.depth.shape[:2]
        xv, yv = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        pixel_coords = np.stack([xv, yv], axis=-1)
        pixel_coords.shape = (H*W, 2)  # view
        points3d = self.camera.inv_project(
            extrinsic, pixel_coords, self.depth.ravel())
        points3d.shape = (H, W, 3)
        return points3d

    @property
    def pc_camera(self):
        """Point cloud in camera frame, returns (H, W, 3)"""
        if self._pc_camera is None:
            self._pc_camera = self._pointcloud(np.eye(4))
        return self._pc_camera
    
    @property
    def pc_world(self):
        """Point cloud in world frame, returns (H, W, 3)"""
        if self._pc_world is None:
            self._pc_world = self._pointcloud(self.extrinsic)
        return self._pc_world


def _build_projection_matrix(intrinsic: CameraIntrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho


