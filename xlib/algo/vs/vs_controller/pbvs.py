import numpy as np

from .vs_controller_base import VisualServoControllerBase
from xlib.algo.utils.transforms import calcPose6dError

class PBVS(VisualServoControllerBase):
    def __init__(self, gain):
        super().__init__()
        self.cur_pose = None
        self.tar_pose = None
        assert gain.shape == (6, ), "gain must be shape (6, )"
        self.K = np.diag(gain)
        
    
    def update(self, cur_pose, tar_pose):
        self.cur_pose = cur_pose
        self.tar_pose = tar_pose
        
    
    def calc_vel(self):
        error_pose = calcPose6dError(self.cur_pose, self.tar_pose)
        vel = self.K @ error_pose
        
        return vel
        
