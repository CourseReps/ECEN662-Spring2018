from math import ceil

import numpy as np
from numpy.random import normal

from aims import *

from aims.attkins import Quat, AngularRate

def int_ceil(val):
    return int(ceil(val))

class SensorSimStatus():

    def __init__(self,rate,sim_time_step,num_sim_time_steps):
        self.time_step_interval = int_ceil((1./rate/sim_time_step))
        self.i = 0
        self.num_measurements = num_sim_time_steps/self.time_step_interval

    def ready_to_update(self,k):
        if k % self.time_step_interval == 0:
            return True
        else:
            return False

def feature_in_view(p_f_c,camera_params):

    false_ret = [False,None]

    x,y,z = p_f_c.flatten()
    if z <=0:
        return false_ret
    x_ratio = x/z
    y_ratio = y/z

    u_max = camera_params.u_max
    v_max = camera_params.v_max
    cx = camera_params.cx
    cy = camera_params.cy
    fx = camera_params.fx
    fy = camera_params.fy

    u = fx*x_ratio + cx
    v = fy*y_ratio + cy

    if u > u_max or u < 0:
        return false_ret
    elif v > v_max or v < 0:
        return false_ret
    else:
        point = PixelFeature(u,v)
        return True, point

def system_sensor_sim(imu_kinematics,features_world,system_params):
    print "Let's simulate some sensors.\n"

    # alias some variable names for notational convenience
    imu_params = system_params.imu_params
    range_params = system_params.range_params
    camera_params = system_params.camera_params
    intersensor_params = system_params.intersensor_params

    # we have simulated kinematics at a discrete number of time steps
    # the sensors will not generally output at each simulation time step
    num_sim_time_steps = imu_kinematics.length()
    sim_time_step = imu_kinematics.time[1]-imu_kinematics.time[0]

    # the status variables hold "bookkeeping info"
    imu_status = SensorSimStatus(imu_params.rate,sim_time_step,num_sim_time_steps)
    range_status = SensorSimStatus(range_params.rate,sim_time_step,num_sim_time_steps)
    feature_status = SensorSimStatus(camera_params.rate,sim_time_step,num_sim_time_steps)

    # these are the sensors we will return
    # just initialize them here
    imu_sensor = IMU(int_ceil(float(num_sim_time_steps)/imu_status.time_step_interval))
    feature_sensor = FeatureDetector(int_ceil(float(num_sim_time_steps)/feature_status.time_step_interval))
    range_sensor = RangeFinder(int_ceil(float(num_sim_time_steps)/range_status.time_step_interval))

    # alias some system parameters for notational convenience
    # these are constant parameters
    p_c_i = intersensor_params.p_c_i
    p_r_i = intersensor_params.p_r_i
    q_ic = intersensor_params.q_ic
    R_ic = q_ic.asRM()
    R_ir = intersensor_params.q_ir.asRM()

    gyro_bias = 0.
    accel_bias = 0.

    # iterate through the simulation time steps
    for k in xrange(num_sim_time_steps):

        # alias more variables for notational convenience
        # these are all the "true" state variables

        # current time
        t_k = imu_kinematics.time[k]

        # position, velocity, and acceleration of imu in world frame
        p_i_w = imu_kinematics.position[:,k:k+1]
        v_i_w = imu_kinematics.velocity[:,k:k+1]
        a_i_w = imu_kinematics.acceleration[:,k:k+1]

        # attitude of imu with respect to world as quaternion...
        q_wi = Quat(imu_kinematics.attitude[:,k:k+1],"xyzw")
        q_wi.normalize()
        # ...and direction cosine matrix
        C_wi = q_wi.asDCM()
        R_wi = C_wi.T

        # body fixed angular velocity
        w_wi = AngularRate(imu_kinematics.angular_rate[:,k:k+1])

        ############################
        # IMU MEASUREMENT SIMULATION
        ############################

        if imu_status.ready_to_update(k):
            imu_measurement = imu_sensor.data[imu_status.i]
            imu_measurement.true_time = t_k
            # imu_measurement.angular_rate = w_wi
            gyro_noise = normal(0,imu_params.gyro_noise,size=(3,1))
            gyro_bias += normal(0,imu_params.gyro_walk,size=(3,1))*1./imu_params.rate
            imu_measurement.angular_rate = w_wi+gyro_bias+gyro_noise
            g = np.array([ [0., 0., -9.79343] ]).T
            accel_bias += normal(0,imu_params.accel_walk,size=(3,1))*1./imu_params.rate
            accel_noise = normal(0,imu_params.accel_noise,size=(3,1))
            imu_measurement.acceleration = np.dot(C_wi,a_i_w-g)+accel_bias+accel_noise
            # imu_measurement.acceleration = np.dot(C_wi,a_i_w-g)
            imu_measurement.seq = imu_status.i
            imu_status.i += 1

        ################################
        # FEATURE MEASUREMENT SIMULATION
        ################################

        if feature_status.ready_to_update(k):
#            print k,num_sim_time_steps,feature_status.i,np.shape(feature_sensor.data)
            feature_measurement = feature_sensor.data[feature_status.i]


            # position of camera in world frame
            p_c_w = p_i_w + np.dot(R_wi, p_c_i)
            # attitude of camera with respect to world frame
            q_wc = q_ic*q_wi
            C_wc = q_wc.asDCM()

            # go through all features
            # see if feature is visible to the camera
            # if it is, find it's projection into camera image
            # probably this could be optimized a lot more
            for n in xrange(features_world.length()):
                # postion of feature in world frame
                p_f_w = features_world[:,n:n+1]

                # position of feature in camera frame
                p_f_c = np.dot(C_wc,p_f_w-p_c_w)

                # is feature visible?
                feature_visible, perfect_projection = feature_in_view(p_f_c,camera_params)

                if feature_visible:
                    perfect_projection.id = n
                    perfect_projection.p_f_w = p_f_w
                    perfect_projection.u += normal(0,camera_params.pixel_noise)
                    perfect_projection.v += normal(0,camera_params.pixel_noise)
                    feature_measurement.features.append(perfect_projection)
            feature_measurement.true_time = t_k
            feature_measurement.seq = feature_status.i
            feature_status.i += 1

        ####################################
        # RANGEFINDER MEASUREMENT SIMULATION
        ####################################

        if range_status.ready_to_update(k):
            range_measurement = range_sensor.data[range_status.i]

            # 33 component of rotation matrix from world frame to rangefinder frame
            R_wr=np.dot(R_wi,R_ir)
            r33 = R_wr[2,2]
            range_measurement.range = -1./r33*(p_i_w[2,0]+np.dot(R_wi[2:3,:],p_r_i)[0,0])
            range_measurement.range += normal(0.,range_params.noise)
            range_measurement.true_time = t_k
            range_measurement.seq = range_status.i
            range_status.i += 1

    return [imu_sensor,feature_sensor,range_sensor]
