
import numpy as np
from aims.attkins import Quat

class EstimatedState(object):

    def __init__(self):
        self.q_wi = Quat()
        self.p_i_w = np.zeros((3,1))
        self.v_i_w = np.zeros((3,1))
        self.b_g = np.zeros((3,1))
        self.b_a = np.zeros((3,1))
        self.mapped_features = []
        self.old_features = []

        self.kf_q_wi = Quat()
        self.kf_p_i_w = np.zeros((3,1))


    def __repr__(self):
        str_repr = ""
        vec_list = [self.p_i_w,self.v_i_w,self.q_wi.asColVector(),
            self.b_g,self.b_a,self.q_wi.asColVector(),self.kf_p_i_w]
        vec_name_list = ["p_i_w","v_i_w","q_wi","gyro_bias","accel_bias","kf_q_wi","kf_p_i_w"]
        for vec,vec_name in zip(vec_list,vec_name_list):
            str_repr += (vec_name + "\t")
            str_repr +=  (str(vec.flatten()) + "\n")
        return str_repr

    # def asVector(self):
    #     return np.vstack((self.))
    #
    #             self.q_wi.asColVector()
    #             self.p_i_w = np.zeros((3,1))
    #             self.v_i_w = np.zeros((3,1))
    #             self.b_g = np.zeros((3,1))
    #             self.b_a = np.zeros((3,1))
    #             self.kf_q_wi = Quat()
    #             self.kf_p_i_w = np.zeros((3,1))
