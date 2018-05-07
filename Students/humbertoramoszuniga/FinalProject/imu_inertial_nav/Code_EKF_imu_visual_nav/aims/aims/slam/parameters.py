
import numpy as np

from aims import *

class EKFParams(Parameters):

    def __init__(self):
        self.system_params = SystemParams()
        self.intialize_radius = 100
        self.observation_radius = 10
        self.gravity = np.array([ [0., 0., -9.81] ]).T
