import numpy as np
from numpy import dot
from matrix_ops import skew_symmetric, col_vector
from axis_angle import AxisAngle


class classproperty(property):
    def __get__(self, cls, owner):
        """
        Normally properties can only be applied to an instance of a class.
        This custom decorator allows the use of "class properties".

        Credit goes to a stack overflow user for the implementation. The
        original developer of this library is not exactly sure how or why this
        implementation works.
        """
        return classmethod(self.fget).__get__(None, owner)()


class Quat(object):
    """
    Represents the quaternion attitude parameterization.
    """

    def __init__(self,quat_as_vector=[0,0,0,1],order="xyzw"):
        """
        Create a quaternion from a sequence of values.

        The sequence must have length 4 and represents the components of the
        imaginary and scalar portions of the quaternion. The sequence can
        be listed in order x,y,z,w or w,x,y,z. x,y,z,w is assumed.
        """

        if order == "xyzw":
            self._data = np.array(quat_as_vector).flatten()
        elif order == "wxyz":
            flatten_q_vec = np.array(quat_as_vector).flatten()
            self._data = np.zeros(4)
            self._data[3] = flatten_q_vec[0]
            self._data[0:3] = flatten_q_vec[1:4]

    @classproperty
    def eye(cls):
        """
        Identity quaternion.
        """
        return Quat([0,0,0,1])

    @classmethod
    def fromAngleAxis(cls,angle,axis):
        """
        Get a quaternion which is a rotation about an arbitray axis.
        """
        from axis_angle import AxisAngle
        return AxisAngle(angle,axis).asQuat()

    @classmethod
    def rand(cls):
        """
        Create a random unit quaternion.
        """
        q_vec = np.random.rand(4)
        q=Quat(q_vec)
        q.normalize()
        return q
    @classmethod
    def fromDCM(cls,C):
        """
        Use Sheppard's algorithm to convert from direction cosine matrix to
        quaternion. See Hurtado, J.E., Kinematic and Kinetic Principles.
        """
        gamma=np.trace(C)
        w2=(1+gamma)/4.
        Ckk=np.diag(C)
        q2=(1+2*Ckk-gamma)/4.
        q2=np.array([q2[0],q2[1],q2[2],w2])

        max_index = np.argmax(q2)
        q=np.zeros(4)
        q[max_index] = np.sqrt(q2[max_index])
        d = 4.*q[max_index]
        C11,C12,C13,C21,C22,C23,C31,C32,C33 = C.flatten()
        if max_index==3:
            q[0] = (C23-C32)/d
            q[1] = (C31-C13)/d
            q[2] = (C12-C21)/d
        elif max_index==0:
            q[3] = (C23-C32)/d
            q[1] = (C12+C21)/d
            q[2] = (C31+C13)/d
        elif max_index==1:
            q[3] = (C31-C13)/d
            q[0] = (C12+C21)/d
            q[2] = (C23+C32)/d
        elif max_index==2:
            q[3] = (C12-C21)/d
            q[0] = (C31+C13)/d
            q[1] = (C23+C32)/d
        quat= Quat(q,order="xyzw")
        quat.normalize()
        return quat

    @property
    def x(self):
        """
        x-component of imaginary portion of quaternion.
        """
        return self._data[0]
    @property
    def y(self):
        """
        y-component of imaginary portion of quaternion.
        """
        return self._data[1]
    @property
    def z(self):
        """
        z-component of imaginary portion of quaternion.
        """
        return self._data[2]
    @property
    def w(self):
        """
        Scalar portion of quaternion.
        """
        return self._data[3]
    @property
    def imaginary(self):
        """
        Imaginary portion of quaternion as 1D array.
        """
        return self._data[0:3]

    def inverse(self):
        """
        Quaternion inverse; see Trawny eq. 13.
        """
        q_vector = np.zeros(4)
        q_vector[:3] = self.imaginary*-1
        q_vector[3] = self.w
        return Quat(q_vector,"xyzw")

    def normalize(self):
        """
        Normalize the quaternion to make a unit quaternion.
        """
        self._data /= self.norm()

    def norm(self):
        """
        The norm of the elements of the quaternion. Should be 1 for a unit
        quaternion.
        """
        return np.sqrt(np.dot(self._data, self._data))

    def Xi(self):
        """
        Matrix that relates angular velocity to quaternion derivative. See
        Trawny eq. 20.
        """
        q = self.imaginary
        q4=self.w

        Q_x=skew_symmetric(q)
        Xi = np.zeros((4,3))
        Xi[:3,:3] = q4*np.eye(3)+Q_x
        Xi[3] = -q
        return Xi

    def Psi(self):
        """
        The "Psi" matrix as defined by Trawny eq. 19.
        """
        q4=self.w
        q=self.imaginary
        q_x = skew_symmetric(q)
        Psi = np.zeros((4,3))
        Psi[:3,:3] = q4*np.eye(3)-q_x
        Psi[3]=-q.T
        return Psi

    def asDCM(self):
        """
        Direction cosine matrix composed of this quaternion. See Trawny eq. 79.
        """
        return np.dot(self.Xi().T,self.Psi())

    def asRM(self):
        """
        Rotation matrix composed of this quaternion.
        """
        return self.asDCM().T

    def asColVector(self,order="xyzw"):
        """
        Column vector representaiton of quaternion. Optional `order` parameter
        defines whether the scalar component is the first or last element of
        the vector.
        """
        if order == "xyzw":
            return col_vector(self._data)
        elif order == "wxyz":
            wxyz_array = np.zeros(4)
            wxyz_array[0] = self._data[3]
            wxyz_array[1:4] = self._data[0:3]
            return col_vector(wxyz_array)


    def __mul__(self, quat2):
        """
        Define quaternion multiplication. See Trawny eq. 10.
        """
        p4=quat2.w
        p = quat2.imaginary
        p_cross = skew_symmetric(p)
        A=np.zeros((4,4))
        A[:3,:3]=p4*np.eye(3)+p_cross
        A[3,0:3] = -p.T
        A[:3,3] = p
        A[3,3] = p4
        quat_as_vector = dot(A,self.asColVector("xyzw"))
        return Quat(quat_as_vector)


    def __repr__(self):
        """
        String representation of Quat.
        """
        str_repr = "[x y z w]=" + str(self._data)
        return str_repr

        # def asAxisAngle(self):
        #
        #     self.normalize()
        #     angle = 2*np.arccos(self.w)
        #     if self.w == 1.:
        #         axis = np.array([1, 1, 1])
        #     else:
        #         axis = self.imaginary / np.sqrt(1-self.w**2)
        #
        #     axis = axis / np.sqrt(np.dot(axis, axis))
        #
        #     return AxisAngle(angle,axis)
