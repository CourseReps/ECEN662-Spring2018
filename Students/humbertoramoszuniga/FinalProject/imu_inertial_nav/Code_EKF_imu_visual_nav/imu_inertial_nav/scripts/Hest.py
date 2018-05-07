import numpy as np


class Hest:
    def __init__(self, vector=np.zeros(3)):
        vector = np.array(vector).flatten()
        self.x = vector[0]
        self.y = vector[1]
        self.z = vector[2]


class Correction:
    def __init__(self, vector=np.zeros(15)):
        vector = np.array(vector).flatten()
        self.data = vector
        self.delq = vector[0:3]
        self.delx = vector[3:6]
        self.delv = vector[6:9]
        self.delbg = vector[9:12]
        self.delba = vector[12:15]

        def __repr__(self):
            """
            String representation of Quat.
            """
            str_repr = "state_error[q p v bg ba]=" + str(self.data)
            return str_repr
