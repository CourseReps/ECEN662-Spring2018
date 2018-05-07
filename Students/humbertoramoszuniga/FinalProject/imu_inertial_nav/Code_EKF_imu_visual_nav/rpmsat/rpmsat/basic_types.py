import numpy as np
from abc import abstractmethod


class NanArray(np.ndarray):
    """
    Numpy array with a few special properties.

    Upon initialization, the array will be filled with nans.

    Data type is float64.
    """
    def __new__(subtype, shape):
        obj = np.ndarray.__new__(subtype, shape, dtype=np.float64)
        obj.fill(np.nan)
        return obj

    def length(self):
        if self.ndim==1:
            return np.shape(self)[0]
        else:
            return np.shape(self)[1]

class XYZWArray(NanArray):
    """
    4xn NanArray.

    The first 4 rows can be addressed using the properties x, y, z
    and w, respectively.
    """
    def __new__(cls, length):
        return NanArray((4,length)).view(cls)

    @property
    def x(self):
        return self[0,:]
    @x.setter
    def x(self,value):
        self[0,:] = value.flatten()
    @property
    def y(self):
        return self[1,:]
    @y.setter
    def y(self,value):
        self[1,:] = value.flatten()
    @property
    def z(self):
        return self[2,:]
    @z.setter
    def z(self,value):
        self[2,:] = value.flatten()
    @property
    def w(self):
        return self[3,:]
    @w.setter
    def w(self,value):
        self[3,:] = value.flatten()

class UVArray(NanArray):
    """
    2xn NanArray.

    The 1st and 2nd rows can be addressed using the properties u and v,
    respectively.
    """
    def __new__(cls, length):
        return NanArray((2,length)).view(cls)

    @property
    def u(self):
        return self[0,:]
    @u.setter
    def u(self,value):
        self[0,:] = value.flatten()
    @property
    def v(self):
        return self[1,:]
    @v.setter
    def v(self,value):
        self[1,:] = value.flatten()

class XYZArray(NanArray):
    """
    3xn NanArray.

    The 1st, 2nd, and 3rd rows can be addressed using the properties x, y, and
    z respectively.
    """
    def __new__(cls, length):
        return NanArray((3,length)).view(cls)

    @property
    def x(self):
        return self[0,:]
    @x.setter
    def x(self,value):
        self[0,:] = value.flatten()
    @property
    def y(self):
        return self[1,:]
    @y.setter
    def y(self,value):
        self[1,:] = value.flatten()
    @property
    def z(self):
        return self[2,:]
    @z.setter
    def z(self,value):
        self[2,:] = value.flatten()

class StateContainer():
    """
    A virtual class that looks nice when "printed."
    """
    def __repr__(self):
        str_repr = ""
        attribute_dict = vars(self)
        for key in attribute_dict:
            str_repr += key + ": \n" + str(attribute_dict[key]) + "\n"
        return str_repr

    # TODO don't think this is enforced
    @abstractmethod
    def length(self):
        pass

class KinematicArray(StateContainer):
    """
    Structure containing position, velocity, acceleration, attitude, and
    angular rate data.
    """
    def __init__(self,length):
        self.time = NanArray(length)
        self.position = XYZArray(length)
        self.velocity = XYZArray(length)
        self.acceleration = XYZArray(length)
        self.attitude = XYZWArray(length)
        self.angular_rate = XYZArray(length)

    def length(self):
        return self.attitude.length()


class PoseArray(StateContainer):
    """
    Structure containing position and attitude data.
    """
    def __init__(self,length):
        self.time = NanArray(length)
        self.position = XYZArray(length)
        self.attitude = XYZWArray(length)

    def length(self):
        return self.attitude.length()

class Pose(StateContainer):
    """
    Structure containing position and attitude data.
    """
    def __init__(self):
        self.time = np.nan
        self.position = NanArray((3,1))
        self.attitude = NanArray((4,1))

    def length(self):
        return self.attitude.length()
