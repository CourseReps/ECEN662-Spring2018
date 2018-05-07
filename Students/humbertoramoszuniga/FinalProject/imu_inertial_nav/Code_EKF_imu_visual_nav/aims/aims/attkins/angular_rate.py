import numpy as np
from matrix_ops import *

class AngularRate(np.ndarray):
    """
    Represents the angular velocity of a body.
    """

    def __new__(cls, rate=np.zeros(3)):
        obj = np.ndarray.__new__(cls, (3,1), dtype=np.float64)
        flattened=np.array(rate).flatten()
        obj[0]=flattened[0]
        obj[1]=flattened[1]
        obj[2]=flattened[2]
        return obj

    def Omega(self):
        Omega = np.zeros((4,4))
        Omega[:3,:3] = -self.skew()
        Omega[:3,3:4] = self
        Omega[3:4,0:3] = -self.T
        return Omega

    def skew(self):
        return skew_symmetric(self)

myW = AngularRate([1,2,3])
