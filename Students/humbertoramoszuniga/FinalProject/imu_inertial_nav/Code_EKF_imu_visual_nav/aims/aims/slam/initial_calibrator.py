
from math import ceil
import numpy as np
from numpy.linalg import multi_dot, inv

from scipy.optimize import minimize

from aims.attkins import Quat,skew_symmetric

def C_from_crp(q):
    q_cross = skew_symmetric(q)
    q2=np.dot(q,q)

    C = np.eye(3)+2./(1+q2)*(-q_cross+np.dot(q_cross,q_cross))

    return C

class InitialCalibrator(object):

    def __init__(self,t_collect,x0,P0,ekf_params):

        t_collect = float(t_collect)
        self.imu_buffer = int(ceil(t_collect*ekf_params.system_params.imu_params.rate))
        self.skip = False
        if self.imu_buffer == 0:
            self.skip = True
        self.s_tilde = np.zeros((self.imu_buffer,3))
        self.w_tilde = np.zeros((self.imu_buffer,3))
        self.range = np.zeros((self.imu_buffer,3))

        self.ekf_params = ekf_params

        self.x0 = x0
        self.P0 = P0

    def register_imu(self,imu_measurement):
        if self.imu_buffer > 0:
            length_imu_buffer = np.shape(self.s_tilde)[0]
            k = length_imu_buffer - self.imu_buffer
            self.s_tilde[k,:] = imu_measurement.acceleration.flatten()
            self.w_tilde[k,:] = imu_measurement.angular_rate.flatten()

            self.imu_buffer -= 1

    def ready(self):
        if self.imu_buffer == 0:
            return True
        else:
            return False


    def cost_function(self,parameters):

        q = parameters[:3]

        #
        C_0i = C_from_crp(q)
        C_wi = self.x0.q_wi.asDCM()

        b_a = np.array([parameters[-3:]]).T

        z_hat = -multi_dot([C_0i,C_wi,self.ekf_params.gravity])+b_a


        residual = self.s_tilde-z_hat.flatten()
        residual = np.array([residual.flatten()]).T

        return np.dot(residual.T,residual).flatten()

    def estimate(self):

        if not self.skip:
            # initial guess on attitude error 
            crp0 = np.array([0,0,0])
            #  intial bias estimate
            b0=self.x0.b_a.flatten()

            # this is vector to optimize
            x0=np.concatenate([crp0,b0])
            # self.cost_function(x0)

            output = minimize(self.cost_function,x0,method="Powell")
            print output


            crp = output.x[:3]

            axis = crp/np.linalg.norm(crp)
            theta = np.arctan(crp[0]/axis[0])*2

            q_0i = Quat.simple_rotation(theta,axis)

            self.x0.q_wi = q_0i*self.x0.q_wi
            self.x0.b_a = np.array([output.x[-3:]]).T


            # g = -self.ekf_params.gravity[2,0]
            # g_I = np.array([ [0,0,g]] )
            # # g_I = np.dot(self.x0.q_wi.asDCM(),self.ekf_params.gravity)
            #
            # alpha3 = np.array([ [0, g], [-g, 0], [0, 0] ])
            # H3 = np.concatenate([alpha3, np.eye(3)],axis=1)
            # # H3 = np.eye(3)
            #
            # n = np.shape(self.s_tilde)[0]
            # H = np.tile(H3, (n,1))
            # z = np.array([ ( self.s_tilde + g_I.flatten() ).flatten() ]).T
            #
            # x_hat = multi_dot([inv(np.dot(H.T,H)),H.T,z])
            # # print x_hat
            #
            # angles = x_hat.flatten()[:2]
            # q3_vec = np.array([angles[0], angles[1], 0.,2])*0.5
            # q=Quat(q3_vec,order="xyzw")
            # q.normalize()
            # self.x0.q_wi = q*self.x0.q_wi
            #
            # self.x0.b_a = x_hat[-3:,:]
            #
            # self.x0.b_g = np.array([ np.mean(self.w_tilde,axis=0) ]).T
