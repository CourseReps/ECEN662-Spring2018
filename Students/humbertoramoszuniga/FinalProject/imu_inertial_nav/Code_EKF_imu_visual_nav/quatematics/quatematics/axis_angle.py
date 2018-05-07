import numpy as np



class AxisAngle(object):

    def __init__(self,angle,axis):

        if type(axis) == str:
            if axis=="x":
                axis = np.array([1,0,0])
            elif axis=="y":
                axis = np.array([0,1,0])
            elif axis=="z":
                axis = np.array([0,0,1])

        self.angle = angle
        self.axis = np.array(axis).flatten()

    def asQuat(self):
        from quat import Quat
        w=np.cos(self.angle/2.)
        normalized_axis = self.axis / np.sqrt(np.dot(self.axis, self.axis))
        q=np.sin(self.angle/2.)*normalized_axis
        q_vec = np.zeros(4)
        q_vec[3] = w
        q_vec[:3] = q
        return Quat(q_vec,"xyzw")

#myAA=AxisAngle(-np.pi,"x")
#print myAA.asQuat()
