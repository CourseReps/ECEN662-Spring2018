import numpy as np
from numpy import pi
#from aims import *
from basic_types import XYZArray
from aims.attkins import Quat

class Parameters(object):

    def __repr__(self):
        str_repr = ""
        attribute_dict = vars(self)
        for key in attribute_dict:
            str_repr += key + ": \n" + str(attribute_dict[key]) + "\n"
        return str_repr

class IMUParams(Parameters):
    def __init__(self):
        self.gyro_noise = 0.001
        self.accel_noise =  0.015
        self.gyro_walk = 0.00004
        self.accel_walk = 0.0002
        self.rate = 200.

class CameraParams(Parameters):

    def __init__(self):
        # self.fx=1144
        # self.fy=1148
        self.fx=432
        self.fy=432
        self.cx=317.
        self.cy=252.
        self.u_max = 640
        self.v_max = 512
        self.rate = 20.
        self.pixel_noise = 0.5
        self.distortion = np.zeros(5)

class RangeParams(Parameters):
    def __init__(self):
        self.rate = 10.
        self.noise = 0.001

class FeatureParams(Parameters):
    def __init__(self):
        self.min_init_confidence = 0.15
        self.max_missed_frames = 10
        self.new_keyframe_threshold = 5

class IntersensorParams(Parameters):

    def __init__(self):
        self.p_c_i=XYZArray(1)
        self.p_r_i=XYZArray(1)
        self.q_ic=Quat([0.5, 0.5,  0.5,  0.5])
        # self.q_ic=Quat.simple_rotation(-pi/2,"x")
        self.p_c_i[:,0] = [0.0508, -0.0508, 0.]
        # self.p_c_i[:,0] = [-0.05, 0., 0.]
        self.p_r_i[:,0] = [0.145, 0.025 , 0.044]
        # self.p_r_i[:,0] = [0, 0.1, -0.1]
        self.q_ir=Quat([0, 0, 0, 1.])
        # self.q_ir=Quat([1,0,0,0])

class SystemParams(Parameters):

    def __init__(self):
        self.imu_params = IMUParams()
        self.range_params = RangeParams()
        self.camera_params = CameraParams()
        self.intersensor_params = IntersensorParams()
        self.feature_params = FeatureParams()
