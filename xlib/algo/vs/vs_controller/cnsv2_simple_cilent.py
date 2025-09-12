import Pyro4
import numpy as np
import Pyro4.naming
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory


# Ref:
# https://stackoverflow.com/questions/77285558/
# why-does-python-shared-memory-implicitly-unlinked-on-exit
def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)

    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        shm_cleanup_fn = resource_tracker._CLEANUP_FUNCS.pop("shared_memory")
    else:
        shm_cleanup_fn = None
    return shm_cleanup_fn


def add_shm_to_resource_tracker(shm_cleanup_fn):
    resource_tracker.register = resource_tracker._resource_tracker.register
    resource_tracker.unregister = resource_tracker._resource_tracker.unregister
    if shm_cleanup_fn is not None:
        resource_tracker._CLEANUP_FUNCS["shared_memory"] = shm_cleanup_fn


def no_track_open_shm(name: str):
    # disable auto unlink operation when the read process exit
    shm_cleanup_fn = remove_shm_from_resource_tracker()
    shm = SharedMemory(name, create=False)
    add_shm_to_resource_tracker(shm_cleanup_fn)
    return shm


class SharedCamera(object):

    DEFAULT_NAME = "CNSv2_m0117_shm_cam"
    DEFAULT_SHAPE = (480, 640, 6)

    def __init__(self, name: str, create: bool, shape: tuple):
        dtype = np.float32
        size = np.prod(shape) * np.dtype(dtype).itemsize

        if create:
            try:
                self.shm = SharedMemory(name=name, create=create, size=size)
            except FileExistsError:
                SharedMemory(name=name, create=False, size=size).unlink()
                self.shm = SharedMemory(name=name, create=create, size=size)
        else:
            self.shm = no_track_open_shm(name)
            assert self.shm.size == size

        self.data_array = np.ndarray(shape=shape, dtype=dtype, buffer=self.shm.buf)

    def write(self, rgbxym: np.ndarray):
        self.data_array[:] = rgbxym
        return self

    def write_rgb(self, rgb: np.ndarray):
        self.data_array[:, :, :3] = rgb

    def write_xy(self, xy: np.ndarray):
        self.data_array[:, :, 3:5] = xy

    def write_mask(self, mask: np.ndarray):
        self.data_array[:, :, 5] = mask

    def read(self, copy: bool = False):
        return self.data_array.copy() if copy else self.data_array


class SimpleClient(object):
    def __init__(self, service_name: str):
        ns: Pyro4.naming.NameServer = Pyro4.locateNS("127.0.0.1", 9090)
        uri = ns.lookup(service_name)
        print("[INFO] Find uri = {}".format(uri))

        self._pyro_proxy = Pyro4.Proxy(uri)
        self._pyro_proxy._pyroSerializer = "pickle"
        self._pyro_proxy._pyroGetMetadata()

        self.shm_cam = SharedCamera(
            name=SharedCamera.DEFAULT_NAME,
            create=False,
            shape=SharedCamera.DEFAULT_SHAPE,
        )

    def set_target(
        self, rgb: np.ndarray, K: np.ndarray, depth_hint: float, mask: np.ndarray = None
    ):
        """
        Args:
            rgb (np.ndarray): shape = (H, W, 3)
            K (np.ndarray): shape = (3, 3)
            depth_hint (float): scene scale
            mask (np.ndarray, optional): shape = (H, W). Defaults to None.
        """
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        xx, yy = np.meshgrid(
            np.arange(rgb.shape[1]), np.arange(rgb.shape[0]), indexing="xy"
        )
        grid = np.stack([xx, yy], axis=-1)  # (H, W, 2)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        norm_xy = (grid - np.array([cx, cy])) / np.array([fx, fy])
        norm_xy = norm_xy.astype(np.float32)

        if mask is None:
            mask = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        else:
            mask = mask.astype(np.float32)

        rgbxym = np.concatenate([rgb, norm_xy, mask[..., None]], axis=-1)  # (H, W, 6)
        self.shm_cam.write(rgbxym)
        self._pyro_proxy.set_target(depth_hint)

    def get_control_rate(
        self, rgb: np.ndarray, mask: np.ndarray = None, mode: str = "auto"
    ):
        """
        Args:
            rgb (np.ndarray): shape = (H, W, 3)
            mask (np.ndarray, optional): shape = (H, W). Defaults to None.
            mode (str): one of ['pbvs', 'hybrid', 'auto']

        Returns:
            vel (np.ndarray): shape = (6,), [v, w]
        """
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        if mask is None:
            mask = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        else:
            mask = mask.astype(np.float32)

        self.shm_cam.write_rgb(rgb)
        self.shm_cam.write_mask(mask)
        vel: np.ndarray = self._pyro_proxy.get_control_rate(mode)
        return vel


def test_client():
    client = SimpleClient("CNSv2_m0117")
    client.set_target(
        rgb=np.random.rand(480, 640, 3),
        K=np.array([[512, 0, 320], [0, 512, 240], [0, 0, 1]]).astype(np.float32),
        depth_hint=0.3,
    )
    vel = client.get_control_rate(rgb=np.random.rand(480, 640, 3), mode="auto")
    print(vel)


if __name__ == "__main__":
    test_client()
