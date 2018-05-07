
import numpy as np
from numpy import dot,sin,cos,cross
from numpy.linalg import multi_dot, norm, inv, eigvals, eig

from aims.attkins import Quat, AngularRate, skew_symmetric
from aims.core import Pose, XYZArray, IMUMeasurement, RangeMeasurement, FeatureMeasurement, ImageMeasurement, GPSMeasurement
from copy import deepcopy
import cv2

from feature_tracker import SimulatedFeatureTracker,Candidate,ImageFeatureTracker

from collections import OrderedDict

class MappedFeature(object):

    def __init__(self,candidate,position,measurement):
        self.descriptor = candidate.query.descriptor
        self.p_f_w = position
        self.prediction = np.zeros((2,1))
        self.measurement = None
        self.uncertainty_radius = 0
        self.id = candidate.id
        self.missed = 0
        self.keyframe = 0

class UsefulSlices(object):
    def __init__(self):
        # these are constant
        self.angle = slice(0,3)
        self.piw = slice(3,6)
        self.viw = slice(6,9)
        self.gyro_bias = slice(9,12)
        self.accel_bias = slice(12,15)
        self.imu_state = slice(0,15)
        self.pose_states = slice(0,6)
        self.imu_kf_states = slice(0,21)

        # initially these have length 0
        self.kf_state = slice(15,15)
        self.kf_piw = slice(15,15)
        self.kf_angle = slice(15,15)
        self.feature_state = slice(0,0)

        # this changes whenever a new feature is added or the keyframe state
        # exists
        self.full_state = slice(0,15)

class StateManager(object):

    def __init__(self):
        self.slices = UsefulSlices()
        self.state_length = 15
        self.num_features = 0
        self.kf_state_on = False
        self.feature_initialized = False
        self.non_feature_length = 15

    def enable_kf_state(self):
        if not self.kf_state_on:
            self.kf_state_on = True
            self.state_length += 6
            self.slices.kf_state = slice(15,21)
            self.slices.kf_angle = slice(15,18)
            self.slices.kf_piw = slice(18,21)
            self.slices.full_state = slice(0,self.state_length)
            self.non_feature_length = 21


    def unregister_feature(self):
        self.state_length -= 3
        self.num_features -= 1
        self.slices.full_state = slice(0,self.state_length)
        if self.kf_state_on:
            self.slices.feature_state = slice(21,21+3*self.num_features)
        else:
            self.slices.feature_state = slice(15,15+3*self.num_features)

    def register_new_feature(self):
        self.feature_initialized = True
        self.state_length += 3
        self.num_features += 1
        self.slices.full_state = slice(0,self.state_length)
        if self.kf_state_on:
            self.slices.feature_state = slice(21,21+3*self.num_features)
        else:
            self.slices.feature_state = slice(15,15+3*self.num_features)



class Propagator(object):
    """
    At time t0, we have angular velocity w0 and acceleration a0.
    At time t1, we have angular velocity wi and acceleration a1.
    At some intermediate time ti, we have angular velocity wi and acceleration
    ai.
    ti = t0 + fract * (t1 - t0)
    Assume:
        wi = w0 + fract * (w1 - w0)
        ai = a0 + fract * (a1 - a0)

    The methods of this function will use these measurements and equations
    to propagate various quantitites in time. All quantitites are a function
    of the argument `fract`. `fract` is the fraction of a single time step.

    Equations and most variable nomenclature comes from [Li & Mourikis, 2012].
    """

    def __init__(self,dt,w_m,s_m,b_g,b_a,q_wi_0,Q,P0,slices):
        """
        Initialize the Propagator.
        """
        self.w_m = w_m
        self.s_m = s_m
        self.b_a = b_a
        self.b_g = b_g.view(type=AngularRate)
        self.dt = dt
        self.C_wi_0 = q_wi_0.asDCM()
        self.Q = Q
        self.P0 = P0
        self.slices = slices

    def rate(self,fract,state):

        # unpack state and covariance
        [x,P] = state

        # quaternion is first part of state
        # it's the rotation from the IMU frame at the last time step
        # to the current IMU frame location
        q_vec=x[0:4,:]
        q_0i=Quat(q_vec)
        q_0i.normalize()
        R_0i=q_0i.asRM()

        # mu is the second part of the state
        mu = x[4:7,0:1]

        # estimated angular rate at this time
        w_hat = self.w_hat(fract)

        # calculate state derivatives
        q_dot=0.5*dot(w_hat.Omega(),q_vec)
        mu_dot = dot(R_0i,self.s_hat(fract))
        rho_dot = mu
        x_dot = np.row_stack((q_dot,mu_dot,rho_dot))

        # DCM that represents rotation from world frame to current IMU frame
        C_wi = dot(q_0i.asDCM(),self.C_wi_0)

        angle_slice = self.slices.angle
        gyro_bias_slice = self.slices.gyro_bias
        accel_bias_slice = self.slices.accel_bias
        viw_slice = self.slices.viw
        piw_slice = self.slices.piw
        imu_slice = self.slices.imu_state
        feature_slice = self.slices.feature_state
        kf_slice = self.slices.kf_state

        # find F, the jacobian of the error state equation.
        F_imu=np.zeros((15,15))
        F_imu[angle_slice,angle_slice] = -skew_symmetric(self.w_hat(fract))
        I3=np.eye(3)
        F_imu[angle_slice,gyro_bias_slice] = -I3
        F_imu[piw_slice,viw_slice] = I3
        F_imu[viw_slice,angle_slice] = -dot(C_wi.T,skew_symmetric(self.s_hat(fract)))
        F_imu[viw_slice,accel_bias_slice] = -C_wi.T

        # find G, which maps noise components onto error state equation
        G_imu = np.zeros((15,12))
        I3 = np.eye(3)
        G_imu[angle_slice,0:3] = -I3
        G_imu[viw_slice,3:6] = -C_wi.T
        G_imu[gyro_bias_slice,6:9] = I3
        G_imu[accel_bias_slice,9:12] = I3

        # find Pdot, time derivative of covariance
        P_imu = P[imu_slice,imu_slice]
        P_dot_imu = dot(F_imu,P_imu)+dot(P_imu,F_imu.T)+multi_dot([G_imu,self.Q,G_imu.T])

        P_imu_feature = P[imu_slice,feature_slice]
        P_dot_imu_feature = dot(F_imu,P_imu_feature)

        P_imu_kf = P[imu_slice,kf_slice]
        P_dot_imu_kf = dot(F_imu,P_imu_kf)

        P_dot = np.zeros(np.shape(P))
        P_dot[imu_slice,imu_slice] = P_dot_imu
        P_dot[imu_slice,feature_slice] = P_dot_imu_feature
        P_dot[feature_slice,imu_slice] = P_dot_imu_feature.T
        P_dot[imu_slice,kf_slice] = P_dot_imu_kf
        P_dot[kf_slice,imu_slice] = P_dot_imu_kf.T

        return (x_dot,P_dot)

    def integrate(self):
        dt=self.dt

        # initial condition for rotation is always no rotation
        q0 = Quat.eye
        q0_vec = q0.asColVector()
        # initial conditions for mu and rho are always 0
        x0=np.row_stack( (q0_vec,np.zeros((6,1))) )
        # initial condition for covariance is current estimate
        P0 = self.P0

        # RK45 propagation
        k1 = self.rate(0,(x0,P0))
        k2 = self.rate(0.5,(x0+dt/2.*k1[0],P0+dt/2.*k1[1]))
        k3 = self.rate(0.5,(x0+dt/2.*k2[0],P0+dt/2.*k2[1]))
        k4 = self.rate(1,(x0+dt*k3[0],P0+dt*k3[1]))
        x1 = x0+dt/6.*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        P1 = P0+dt/6.*(k1[1]+2*k2[1]+2*k3[1]+k4[1])

        # unpack quaternion from propagated state vector
        q1_vec=x1[:4,0:1]
        q1 = Quat(q1_vec)
        q1.normalize()

        # unpack mu and rho from propagated state vector
        mu1 = x1[4:7,0:1]
        rho1 = x1[7:10,0:1]

        return [q1,mu1,rho1,P1]

    def w_hat(self,fract):
        """
        Returns the body fixed angular velocity estimate at time ti.
        """
        w0 = self.w_m[0]
        w1 = self.w_m[1]
        return fract*(w1-w0)+w0-self.b_g

    def s_hat(self,fract):
        """
        Returns the body fixed acceleration estimate at time ti.
        """
        s0 = self.s_m[0]
        s1 = self.s_m[1]
        return fract*(s1-s0)+s0-self.b_a



def least_squares_jacobian(optimize_variables,Jf_i,C_nj,p_n_j):
  """
  Calculates the Jacobian of the function we are trying to minimize with
  nonlinear least squares.

  This is a very inelegant implementation. I calculated the Jacobian with
  MATLAB and basically copied and pasted.
  """
  alpha, beta, rho = optimize_variables.flatten()
  c11,c12,c13,c21,c22,c23,c31,c32,c33 = C_nj.flatten()
  p1,p2,p3 = p_n_j.flatten()

  Jf_i[0,0] = c11/(c33 + alpha*c31 + beta*c32 + p3*rho) - (c31*(c13 + alpha*c11 + beta*c12 + p1*rho))/(c33 + alpha*c31 + beta*c32 + p3*rho)**2
  Jf_i[0,1] = c12/(c33 + alpha*c31 + beta*c32 + p3*rho) - (c32*(c13 + alpha*c11 + beta*c12 + p1*rho))/(c33 + alpha*c31 + beta*c32 + p3*rho)**2
  Jf_i[0,2] = p1/(c33 + alpha*c31 + beta*c32 + p3*rho) - (p3*(c13 + alpha*c11 + beta*c12 + p1*rho))/(c33 + alpha*c31 + beta*c32 + p3*rho)**2
  Jf_i[1,0] = c21/(c33 + alpha*c31 + beta*c32 + p3*rho) - (c31*(c23 + alpha*c21 + beta*c22 + p2*rho))/(c33 + alpha*c31 + beta*c32 + p3*rho)**2
  Jf_i[1,1] = c22/(c33 + alpha*c31 + beta*c32 + p3*rho) - (c32*(c23 + alpha*c21 + beta*c22 + p2*rho))/(c33 + alpha*c31 + beta*c32 + p3*rho)**2
  Jf_i[1,2] = p2/(c33 + alpha*c31 + beta*c32 + p3*rho) - (p3*(c23 + alpha*c21 + beta*c22 + p2*rho))/(c33 + alpha*c31 + beta*c32 + p3*rho)**2

class Interpolator(object):

    def __init__(self,time,w,a):
        self.time = deepcopy(time)
        self.w = deepcopy( w )
        self.a = deepcopy(a)

    def interp(self,t):
        fract = (t-self.time[0])/(self.time[1]-self.time[0])
        w0 = self.w[0]
        w1 = self.w[1]
        w_t = fract*(w1-w0)+w0
        a0 = self.a[0]
        a1 = self.a[1]
        a_t = fract*(a1-a0)+a0
        return w_t,a_t

class EKFSLAM(object):

    def __init__(self, initial_state,P0,ekf_params,clock):
        self.state = deepcopy(initial_state)

        self.k = 0
        self.last_update_t = clock.now()
        self.clock = clock

        self.visible_features = []

        # set initial covariance
        self.P = deepcopy(P0)

        # the manager keeps track of references to different parts of the state
        self.manager = StateManager()

        # save filter parameters as different variables
        ekf_params = deepcopy(ekf_params)
        self.ekf_params = ekf_params
        self.intersensor_params = ekf_params.system_params.intersensor_params
        self.camera_params = ekf_params.system_params.camera_params
        self.range_params = ekf_params.system_params.range_params
        self.feature_params = ekf_params.system_params.feature_params

        # fill in the process noise matrix Q
        imu_params = ekf_params.system_params.imu_params
        s_n_g = [imu_params.gyro_noise]*3
        s_n_a = [imu_params.accel_noise]*3
        s_n_gb = [imu_params.gyro_walk]*3
        s_n_ab = [imu_params.accel_walk]*3
        Q_diag = np.concatenate([s_n_g,s_n_a,s_n_gb,s_n_ab])**2
        self.Q=np.diag(Q_diag)

        # set the camera matrix
        cx = self.camera_params.cx
        cy = self.camera_params.cy
        fx = self.camera_params.fx
        fy = self.camera_params.fy
        u_max = self.camera_params.u_max
        v_max = self.camera_params.v_max
        self.camera_matrix = np.array([ [fx, 0,  cx],
                                        [ 0, fy, cy],
                                        [ 0,  0,  1] ])

        # this will do feature tracking from monocular camera images
        # used with camera_update()
        self.image_feature_tracker = ImageFeatureTracker(self.feature_params.new_keyframe_threshold)

        # in the case that the feature measurements are made externally,
        # this will mimic the action of image_feature_tracker
        # used with feature_update()
        self.sim_feature_tracker = SimulatedFeatureTracker((u_max,v_max),self.feature_params.new_keyframe_threshold)

        self.measure_queue = []

        self.ratio = np.nan

    def add_measurement(self,new_measurement):

        # if this is an IMU mesurement
        if isinstance(new_measurement,IMUMeasurement):
            # latest IMU measurements
            w_m_1 = new_measurement.angular_rate
            a_m_1 = new_measurement.acceleration

            # we need the (k-1) imu measurement in order to propagate
            if self.k != 0:
                self.measure_queue.append(new_measurement)

                # the interpolator will provide interpolated IMU
                # measurements at any time between the last IMU
                # measurement and the current IMU measurement
                times = (self.last_update_t,new_measurement.true_time)
                w = (self.w_m_0,w_m_1)
                a = (self.a_m_0,a_m_1)
                interp = Interpolator(times,w,a)

                sorted_queue = sorted(self.measure_queue, key = lambda x:x.true_time)
                for measure in sorted_queue:
                    # propagate between measurements
                    w0,a0 = interp.interp(self.last_update_t)
                    w1,a1 = interp.interp(measure.true_time)
                    dt = measure.true_time-self.last_update_t
                    self.time_update((w0,w1),(a0,a1),dt)

                    # if not an IMU measurement, call the appropriate update
                    # function
                    if isinstance(measure,RangeMeasurement):
                        self.range_update(measure)
                    elif isinstance(measure,FeatureMeasurement):
                        self.feature_update(measure)
                    elif isinstance(measure,ImageMeasurement):
                        self.camera_update(measure)
                    elif isinstance(measure,GPSMeasurement):
                        self.gps_update(measure)

                    self.last_update_t = measure.true_time

            # if this is the first IMU measurement
            elif self.k == 0:
                self.last_update_t = new_measurement.true_time

            # reset measurement queue
            self.measure_queue = []
            # update stored IMU measurements for next time
            self.w_m_0 = w_m_1
            self.a_m_0 = a_m_1
            self.k += 1
        else:
            self.measure_queue.append(new_measurement)

    def camera_pose(self,p_i_w=None,q_wi=None):
        """
        Returns the pose of the camera given the pose of the IMU. If no
        IMU pose is given, the current IMU pose is used.
        """
        if p_i_w==None:
            p_i_w = self.state.p_i_w
        if q_wi==None:
            q_wi = self.state.q_wi
        # constant rotation from IMU to camera
        q_ic = self.intersensor_params.q_ic
        # constant position of camera in IMU frame
        p_c_i = self.intersensor_params.p_c_i

        # attitude of camera in world frame
        q_wc =  q_ic*q_wi
        # position of camera in world frame
        p_c_w = p_i_w + dot(q_wi.asRM(),p_c_i)

        camera_pose = Pose()
        camera_pose.position = p_c_w
        camera_pose.attitude = q_wc
        return camera_pose


    def _augment_state_with_feature(self,Pxx,Pxp):
        """
        Update the covariance with the feature covariance and associated
        cross correlation terms.
        """

        # new covariance is the old covariance but with 3 new rows and columns
        # append to the end
        # print Pxx
        # print Pxp
        self.manager.register_new_feature()
        newP = np.zeros((self.manager.state_length,self.manager.state_length))
        n_p = self.P.shape[0]
        newP[:n_p,:n_p] = self.P
        newP[-3:,:n_p] = Pxp
        newP[:n_p,-3:] = Pxp.T
        # newP[-3:,:self.state_length_now] = Pxp
        # newP[:self.state_length_now,-3:] = Pxp.T
        newP[-3:,-3:] = Pxx
        self.P = newP
        # print self.P
        # dd

    def _augment_state_with_imu_pose(self):

        slices = self.manager.slices

        # if we don't have a past pose state...
        if not self.manager.kf_state_on:
            self.manager.enable_kf_state()
            n_p = self.P.shape[0]
            # create new covariance matrix with room for the pose
            newP = np.zeros((self.manager.state_length,self.manager.state_length))
            # copy the old covariance into the new covariance
            newP[:n_p,:n_p] = self.P
            # set the pose covariance, which equals the current pose covariance
            newP[slices.kf_state,slices.kf_state] = self.P[slices.pose_states,slices.pose_states]
            # set the other terms
            newP[slices.kf_state,:n_p] = self.P[slices.pose_states,:]
            newP[:n_p,slices.kf_state] = self.P[:,slices.pose_states]
            self.P = newP
        else:
            self.P[slices.kf_state,:] = self.P[slices.pose_states,:]
            self.P[:,slices.kf_state] = self.P[:,slices.pose_states]
            self.P[slices.kf_state,slices.kf_state] = self.P[slices.pose_states,slices.pose_states]


        # augment state
        self.state.kf_p_i_w = deepcopy(self.state.p_i_w)
        self.state.kf_q_wi = deepcopy(self.state.q_wi)

    def pixel_to_ratio(self,uv):
        """
        Convert a pixel measurement to triangle ratios.
        """

        u_m = uv[0]
        v_m = uv[1]

        # camera parameters
        cx = self.camera_params.cx
        cy = self.camera_params.cy
        fx = self.camera_params.fx
        fy = self.camera_params.fy

        z_m = np.zeros((2,1))
        z_m[0,0] = (u_m-cx)/fx
        z_m[1,0] = (v_m-cy)/fy

        return z_m

    def _Jh_p(self):
        """
        The jacobian of h with respect to the IMU position, where
        h is the position of the feature in the camera frame.
        """
        return dot(-self.intersensor_params.q_ic.asDCM(),self.state.q_wi.asDCM())

    def _Jh_ang(self,p_f_w):
        """
        The jacobian of h with respect to attitude error, where
        h is the position of the feature in the camera frame.
        """
        C_ic = self.intersensor_params.q_ic.asDCM()
        C_wi = self.state.q_wi.asDCM()
        p_i_w = self.state.p_i_w
        return dot(C_ic,skew_symmetric(dot(C_wi,p_f_w-p_i_w) ))

    def _h_c(self,camera_pose,p_fi_w):
        """
        Returns the Euclidean position of a feature in the camera frame given
        the global camera pose and the global feature position.
        """

        return dot(camera_pose.attitude.asDCM(),p_fi_w-camera_pose.position)

    def _Jyh(self,h):
        """
        The jacobian of the measurement equation with respect to h, where
        h is the position of the feature in the camera frame.
        """
        hx,hy,hz = h.flatten()

        # camera parameters
        cx = self.camera_params.cx
        cy = self.camera_params.cy
        fx = self.camera_params.fx
        fy = self.camera_params.fy

        return np.array([ [fx, 0., -fx*hx/hz],
                          [0., fy, -fy*hy/hz] ])*1./hz

    def _yc(self,h):
        """
        The projected pixel measurement of a feature given h, where
        h is the position of the feature in the camera frame.
        """
        hx,hy,hz = h.flatten()

        # camera parameters
        cx = self.camera_params.cx
        cy = self.camera_params.cy
        fx = self.camera_params.fx
        fy = self.camera_params.fy

        return np.array([ [fx*hx/hz+cx], [fy*hy/hz+cy] ])

    def _init_candidate_pair(self,candidate):

        uv1=candidate.query.pt
        uv2=candidate.train.pt

        cam_pose_1 = self.camera_pose(self.state.kf_p_i_w,self.state.kf_q_wi)
        cam_pose_2 = self.camera_pose(self.state.p_i_w,self.state.q_wi)

        residual = np.ones((4,1))
        # initialize Jacobian matrix
        Jf = np.zeros((4,3))
        tol = 1e-4
        newton_iteration = 0
        delta_optimize_variables = 100

        optimize_variables = np.ones((3,1))*0.3
        optimize_variables[2,0] = 1.

        z1 = self.pixel_to_ratio(uv1)
        z2 = self.pixel_to_ratio(uv2)

        max_iter = 10

        while dot(residual.T,residual) >= tol and newton_iteration < max_iter:
            alpha, beta, rho = optimize_variables.flatten()

            # find measurement Jacobian and residual for initial observation
            # measurement Jacobian
            # this probably doesn't require a call since it's probably constant
            least_squares_jacobian(optimize_variables,Jf[:2,:],np.eye(3),np.zeros((3,1)))

            # estimated measurement
            H1 = np.array([ [alpha,beta,1.] ]).T
            z1_est = 1/H1[2,0] * H1[:2]
            residual[:2,:] = z1-z1_est

            # find measurement Jacobian and residual for 2nd observation

            # rotation of camera between first observation and
            # this observation
            q_12 = cam_pose_2.attitude*(cam_pose_1.attitude.inverse())
            C12 = q_12.asDCM()

            # distance to camera frame at first observation,
            # in camera frame at this observation
            p_1_2 = dot(cam_pose_2.attitude.asDCM(),cam_pose_1.position - cam_pose_2.position)
            least_squares_jacobian(optimize_variables,Jf[2:4,:],C12,p_1_2)

            H2 = (dot(C12,np.array([ [alpha,beta,1.] ]).T) +
               rho*p_1_2)
            z2_est = 1/H2[2,0] * H2[:2]
            residual[2:4,:] = z2-z2_est

            # guass-newton parameter update
            try:
                new_optimize_variables = (optimize_variables +
                            multi_dot([inv(dot(Jf.T,Jf)),Jf.T,residual]))
            except:
                return
            delta_optimize_variables = norm(new_optimize_variables -
                                optimize_variables)
            optimize_variables = new_optimize_variables
            newton_iteration += 1

        # print dot(residual.T,residual)
        if dot(residual.T,residual) <= tol:

            alpha, beta, rho = optimize_variables.flatten()
            p_f_w = 1./rho * dot(cam_pose_1.attitude.asDCM().T,np.array([ [alpha,beta,1.] ]).T) + cam_pose_1.position


            # redefine variables for clarity
            p_i_w_1 = self.state.kf_p_i_w
            p_i_w_2 = self.state.p_i_w

            C_wi_1 = self.state.kf_q_wi.asDCM()
            C_wi_2 = self.state.q_wi.asDCM()

            C_wc_1 = cam_pose_1.attitude.asDCM()
            C_wc_2 = cam_pose_2.attitude.asDCM()

            C_ic = self.intersensor_params.q_ic.asDCM()

            hc_1 = self._h_c(cam_pose_1,p_f_w)
            hc_2 = self._h_c(cam_pose_2,p_f_w)

            Jyh_1 = self._Jyh(hc_1)
            Jyh_2 = self._Jyh(hc_2)

            Hx_1 = multi_dot([Jyh_1,C_wc_1])
            Hx_2 = multi_dot([Jyh_2,C_wc_2])
            Hx = np.vstack((Hx_1,Hx_2))


            slices = self.manager.slices

            # Hp_1 = np.zeros((2,self.state_length_now))
            Hp_1 = np.zeros((2,self.manager.state_length))
#            print Hp_1[:,slices.kf_angle]
            Hp_1[:,slices.kf_angle] = multi_dot([
                                        Jyh_1,
                                        C_ic,
                                        skew_symmetric(
                                             dot(C_wi_1,
                                                  p_f_w-p_i_w_1)
                                             )
                                        ])
            Hp_1[:,slices.kf_piw] = -dot(Jyh_1,C_wc_1)

            # Hp_2 = np.zeros((2,self.state_length_now))
            Hp_2 = np.zeros((2,self.manager.state_length))
            Hp_2[:,slices.angle] = multi_dot([
                                        Jyh_2,
                                        C_ic,
                                        skew_symmetric(
                                             dot(C_wi_2,
                                                  p_f_w-p_i_w_2)
                                             )
                                        ])
            Hp_2[:,slices.piw] = -dot(Jyh_2,C_wc_2)
            Hp = np.vstack((Hp_1,Hp_2))

            Ppp_inv = inv(self.P)
            R = np.diag(np.concatenate((candidate.query.var,candidate.train.var)))

            W=inv(R)
            P1 = inv(multi_dot([Hp.T,W,Hp])+Ppp_inv)
            try:
                Pxx = inv( multi_dot([Hx.T,W,Hx])
                        -multi_dot([Hx.T,W,Hp,P1,Hp.T,W,Hx])  )
            except:
                return

            Pxp = -multi_dot([Pxx,Hx.T,W,Hp,P1])

            try:
                var, Rot = eig(Pxx)
            except:
                return
            v = eigvals(Pxx).flatten()
            conf = np.sqrt(np.max(v))*3
            # ratio = conf/rho

            if (conf <= self.feature_params.min_init_confidence and rho > 0):
                    self.state.mapped_features.append(MappedFeature(candidate,p_f_w,uv2))
                    self._augment_state_with_feature(Pxx,Pxp)
            return conf
        return None
        # else:
        #     print "fail"


    def feature_covariance(self):
        """
        Returns the 3m x 3m submatrix representing the covariance of the
        feature states.
        """
        slices = self.manager.slices
        return self.P[slices.feature_state,slices.feature_state]

    def three_sigma(self):
        return  np.array([ np.sqrt(np.diag(self.P))*3. ]).T


    def camera_update(self,camera_measurement):
        """
        Update the filter using a monocular camera image.
        """
        # for feature in self.state.mapped_features:
        #     feature.measurement = None
        self._predict_features(self.state.mapped_features,True)
        undistorted = cv2.undistort(camera_measurement.image,self.camera_matrix,self.camera_params.distortion)
        candidates, new_kf = self.image_feature_tracker.register_image(
            undistorted, self.state.mapped_features,self.camera_pose())

        self._process_candidates(candidates,new_kf)


    def _predict_features(self,some_mapped_features,find_uncertainty=False,find_residual=False):

        num_features = len(some_mapped_features)

        if num_features > 0:
            length_residual = 2*num_features
            Hc = np.zeros((length_residual,self.manager.state_length))
            residual = np.zeros((length_residual,1))
            slices = self.manager.slices

            camera_pose = self.camera_pose()
            Jh_p = self._Jh_p()
            i=0
            for feature in some_mapped_features:
                p_f_w = feature.p_f_w
                hc = self._h_c(camera_pose,p_f_w)
                feature.prediction = self._yc(hc)

                row_slice = slice(2*i,2*(i+1))
                L = self.manager.non_feature_length
                feature_slice = slice(L+3*i,L+3*(i+1))
                Hc_i = Hc[row_slice,:]

                if find_residual:
                    residual_i = residual[row_slice,:]
                    residual_i[:] = feature.measurement - feature.prediction


                Jyh = self._Jyh(hc)

                Jh_ang = self._Jh_ang(p_f_w)
                Hc_i[:,slices.angle] = dot(Jyh,Jh_ang)
                Hc_i[:,slices.piw] = dot(Jyh,Jh_p)
                Hc_i[:,feature_slice] = -dot(Jyh,Jh_p)

                i+=1

            # if find_uncertainty:
            #     R=np.eye(length_residual)*self.camera_params.pixel_noise**2
            #     Pyy = multi_dot([Hc,self.P,Hc.T])+R
            #     for i, feature in enumerate(self.state.mapped_features):
            #         feature_slice = slice(2*i,2*(i+1))
            #         Pff = Pyy[feature_slice,feature_slice]
            #         feature.uncertainty_radius = np.sqrt(np.max(eigvals(Pff)))*3
                    # feature.uncertainty_radius = 15


            return Hc,residual

        return None

    def feature_update(self,feature_measurement):
        """
        Update the filter using direct pixel measurements of features.
        This is useful if external methods have already been used to extract
        features.
        """
        self._predict_features(self.state.mapped_features)
        for feature in self.state.mapped_features:
            feature.measurement = None
        candidates, new_kf = self.sim_feature_tracker.register_features(
        feature_measurement, self.state.mapped_features)
        self._process_candidates(candidates,new_kf)

    def _process_candidates(self,candidates, new_kf):

        # augment state with keyframe if a new keyframe was declared
        if new_kf:
            self._augment_state_with_imu_pose()

        # list of feature indices that we have measurements for
        self.visible_features = [feature for feature in
                            self.state.mapped_features if
                            feature.measurement != None]

        # if there is "sufficient" baseline, try to initialize feature
        conf_sum = 0.
        conf_num = 0
        for candidate in candidates:
            p1 = candidate.query.pt
            p2 = candidate.train.pt
            pixel_baseline = norm(p1-p2)
            if pixel_baseline >= 25:
                conf = self._init_candidate_pair(candidate)
                if conf != None:
                    conf_sum += conf
                    conf_num += 1
        if conf_num > 0:
            self.ratio = conf_sum/conf_num



        # update state using observations of mapped features
        if len(self.visible_features) > 0:
            # print "%i/%i features visible" %(len(self.visible_features),self.manager.num_features)
            Hc, residual = self._predict_features(self.visible_features,False,True)
            R=np.eye(len(self.visible_features)*2)*self.camera_params.pixel_noise**2
            self._kalman_update(R,Hc,residual)

    def _remove_feature(self,feature_id):
        self.state.old_features.append(self.state.mapped_features.pop(feature_id))
        L = self.manager.non_feature_length
        ref =  L+3*feature_id
        refs = (ref,ref+1,ref+2)
        self.P = np.delete(self.P,refs,axis=0)
        self.P = np.delete(self.P,refs,axis=1)
        self.manager.unregister_feature()



    def _update_attitude(self,alpha,q):
        """
        Update quaternion estimate using a "small angle".
        """
        Xi = q.Xi()
        q_vec = q.asColVector(order="xyzw")
        delta_q = 0.5*dot(Xi,alpha)
        q = Quat(q_vec+delta_q)
        q.normalize()
        return q

    def _kalman_update(self,R,H,residual):
        """
        Calculate Kalman gain and use it to update all filter states.
        """

        # calculate Kalman gain
        Pyy = multi_dot([H,self.P,H.T])+R
        K=multi_dot([self.P,H.T,inv(Pyy)])

        # update covariance
        self.P -= multi_dot([K,H,self.P])

        slices = self.manager.slices
        # update the states with additive error
        delta_x = dot(K,residual)
        # print delta_x
        self.state.p_i_w += delta_x[slices.piw,:]
        self.state.v_i_w += delta_x[slices.viw,:]
        self.state.b_g += delta_x[slices.gyro_bias,:]
        self.state.b_a += delta_x[slices.accel_bias,:]
        # update attitude with multiplicative error
        self.state.q_wi = self._update_attitude(delta_x[slices.angle,:],self.state.q_wi)

        if self.manager.kf_state_on:
            self.state.kf_p_i_w += delta_x[slices.kf_piw,:]
            self.state.kf_q_wi = self._update_attitude(delta_x[slices.kf_angle,:],self.state.kf_q_wi)

        u_max = self.camera_params.u_max
        v_max = self.camera_params.v_max
        delta = 50
        u_lims = (-delta,u_max + delta)
        v_lims = (-delta,v_max + delta)
        L = self.manager.non_feature_length
        for i,feature in enumerate(self.state.mapped_features):
            feature_slice = slice(L+3*i,L+3*(i+1))
            feature.p_f_w += delta_x[feature_slice,:]
            u = feature.prediction[0,0]
            v = feature.prediction[1,0]
            if u < u_lims[0] or u > u_lims[1] or v < v_lims[0] or v > v_lims[1]:
                self._remove_feature(i)
            if feature.missed > self.feature_params.max_missed_frames:
                self._remove_feature(i)

    def gps_update(self,gps_measurement):

        yhat = self.state.p_i_w
        residual = gps_measurement.position-yhat
        R=(0.01/3.)**2*np.eye(3)

        slices = self.manager.slices
        H = np.zeros((3,self.manager.state_length))

        H[:,slices.piw] = np.eye(3)

        self._kalman_update(R,H,residual)



    def range_update(self,range_measurement):
        """
        Update the filter using the range measurement.
        """

        # third row of rotation matrix from world to imu
        R_wi = self.state.q_wi.asRM()
        R_wi_3j = R_wi[2:3,:]

        # 33 component of rotation matrix from world frame to rangefinder frame
        R_ir = self.intersensor_params.q_ir.asRM()
        R_wr=dot(R_wi,R_ir)
        r33 = R_wr[2,2]

        # third component of position of imu in world frame
        p_iw_3 = self.state.p_i_w[2:3,:]

        # 33 position of rangefinder in world frame
        p_r_i = self.intersensor_params.p_r_i
        p_r_w_3 = p_iw_3+dot(R_wi_3j,p_r_i)

        # estimated range measurement
        yhat = -1./r33*p_r_w_3

        # measurement residual
        residual = range_measurement.range-yhat
        # print residual,range_measurement.true_time,range_measurement.range

        # calculate jacobian of measurement with respect to attitude error
        R_ir_i3_cross = skew_symmetric(R_ir[:,2])
        R_wi_3j_cross = skew_symmetric(R_wi_3j)
        Hr_theta = (1./r33**2*p_r_w_3[0]*dot(-R_wi_3j,R_ir_i3_cross) -
                    1./r33*dot(p_r_i.T,R_wi_3j_cross))

        # calculate jacobian with respect to third component of IMU position
        Hr_p = -1./r33

        # full jacobian
        slices = self.manager.slices
        Hr=np.zeros((1,self.manager.state_length))
        Hr[:,slices.angle] = Hr_theta
        Hr[:,5] = Hr_p

        # update state using residual and jacobian
        R=self.range_params.noise**2*np.ones((1,1))

        self._kalman_update(R,Hr,residual)

    def time_update(self,w,a,dt):

        # remap state variables for brevity
        b_a_0 = self.state.b_a
        b_g_0 = self.state.b_g
        q_wi_0 = self.state.q_wi
        R_wi_0 = q_wi_0.asRM()
        v_i_w_0 = self.state.v_i_w
        p_i_w_0 = self.state.p_i_w
        P0 = self.P

        # set up the propagator
        # this will solve the system of partial differential equations
        # for the states and covariance
        pr = Propagator(dt,w,a,
                        b_g_0,b_a_0,q_wi_0,self.Q,P0,self.manager.slices)

        # 4 variables are integrated
        # these will be used to calculate the new states
        q1,mu1,rho1,self.P = pr.integrate()

        # update attitude
        self.state.q_wi = q1*q_wi_0

        # update velocity
        g_w = self.ekf_params.gravity
        self.state.v_i_w = v_i_w_0 + dot(R_wi_0,mu1) + g_w*dt

        # update position
        self.state.p_i_w = (p_i_w_0 + v_i_w_0*dt + dot(R_wi_0,rho1)
                                + 0.5*g_w*dt**2)
