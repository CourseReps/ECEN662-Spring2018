import numpy as np
from matrix_ops import *

class AngularRate(np.ndarray):
    """
    Represents the angular velocity of a body.

    AngularRate is an inherited 3 x 1 numpy column vector. It represents
    the angular velocity vector.
    """

    def __new__(cls, rate=np.zeros(3)):
        obj = np.ndarray.__new__(cls, (3,1), dtype=np.float64)
        flattened=np.array(rate).flatten()
        obj[0]=flattened[0]
        obj[1]=flattened[1]
        obj[2]=flattened[2]
        return obj

    def Omega(self):
        """
        See Trawny, equation 107, for the primary use of the Omega matrix.
        """
        Omega = np.zeros((4,4))
        Omega[:3,:3] = -self.skew()
        Omega[:3,3:4] = self
        Omega[3:4,0:3] = -self.T
        return Omega

    def skew(self):
        """
        Returns the angular velocity matrix i.e. the skew symmetric matrix
        formed from the angular velocity vector. This is frequently useful.
        """
        return skew_symmetric(self)
