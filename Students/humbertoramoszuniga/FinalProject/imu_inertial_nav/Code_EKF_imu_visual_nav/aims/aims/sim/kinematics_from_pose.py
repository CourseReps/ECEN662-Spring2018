


from math import ceil

import numpy as np
import matplotlib.pylab as plt
from multiplot2d import MultiPlotter

from aims import *

from aims.attkins import Quat, AngularRate

from scipy.linalg import expm

from numpy import dot

# import pickle

plt.close("all")
#plt.style.use("dwplot")

def five_point_stencil(data,h):
    """
    Calculate numerical derivative using 5 point stencil.
    """
    length=np.shape(data)[0]

    derivative = np.zeros(length)
    for k in xrange(2,length-2):
        derivative[k] = (-data[k+2]+8*data[k+1]-8*data[k-1]+data[k-2])/12/h
    return derivative[2:-2]

def xyz_five_point_stencil(xyz_data,h):
    derivative = XYZArray(xyz_data.length()-4)
    derivative.x = five_point_stencil(xyz_data.x,h)
    derivative.y = five_point_stencil(xyz_data.y,h)
    derivative.z = five_point_stencil(xyz_data.z,h)
    return derivative

def xyzw_five_point_stencil(xyzw_data,h):
    derivative = XYZWArray(xyzw_data.length()-4)
    derivative.x = five_point_stencil(xyzw_data.x,h)
    derivative.y = five_point_stencil(xyzw_data.y,h)
    derivative.z = five_point_stencil(xyzw_data.z,h)
    derivative.w = five_point_stencil(xyzw_data.w,h)
    return derivative

def xyz_interp(interp_time,original_time,xyz_data):
    interp_data = XYZArray(len(interp_time))
    interp_data.x = np.interp(interp_time,original_time,xyz_data.x)
    interp_data.y = np.interp(interp_time,original_time,xyz_data.y)
    interp_data.z = np.interp(interp_time,original_time,xyz_data.z)
    return interp_data

def xyzw_interp(interp_time,original_time,xyzw_data):
    interp_data = XYZWArray(len(interp_time))
    interp_data.x = np.interp(interp_time,original_time,xyzw_data.x)
    interp_data.y = np.interp(interp_time,original_time,xyzw_data.y)
    interp_data.z = np.interp(interp_time,original_time,xyzw_data.z)
    interp_data.w = np.interp(interp_time,original_time,xyzw_data.w)
    return interp_data

def kinematics_from_pose(original_pose,dt_interp=0.1,dt_new=0.005):
    """
    Take pose data and return an approximation of the kinematic history.

    The pose data time step does not have to be constant.

    """

    if not np.isclose(0.,dt_interp%dt_new):
        raise ValueError("dt_interp must be multiple of dt_new.")
    elif dt_new > dt_interp:
        raise ValueError("dt_new must be smaller than dt_interp")

    # linearlly interpolate the original position

    # create vector of times at which we want to interpolate position
    original_final_time = original_pose.time[-1]
    interp_time = np.arange(0,original_final_time,dt_interp)
    # linearlly interpolate the position
    interp_pos = xyz_interp(interp_time,original_pose.time,original_pose.position)

    # find numerical derivative of position...
    numerical_velocity = xyz_five_point_stencil(interp_pos,dt_interp)
    # ...and velocity
    numerical_accel = xyz_five_point_stencil(numerical_velocity,dt_interp)

    # interpolate numerical acceleration

    # create new time vector
    # with five point stencil, we cannot estimate derivatives for entire range
    # since we took two derivatives, we lose 4 points on each "end" of the
    # original dataset
    numerical_accel_times = interp_time[4:-4]
    # create evenly spaced grid of times
    new_state_length = int(ceil((numerical_accel_times[-1]-numerical_accel_times[0])/dt_new))
    new_state = KinematicArray(new_state_length)
    new_state.time[:] = np.arange(numerical_accel_times[0],numerical_accel_times[-1],dt_new)
    # interpolate numerical acceleration
    # we are assuming 1st order hold on acceleration
    # so we are really just adding "resolution" to our state history
    new_state.acceleration = xyz_interp(new_state.time,numerical_accel_times,numerical_accel)

    # double integrate numerical acceleration

    # create views of data for notational convenience
    # no copies here
    a=new_state.acceleration
    v=new_state.velocity
    p=new_state.position
    # set initial conditions
    v[:,0] = numerical_velocity[:,2]
    p[:,0] = interp_pos[:,4]
    for k in xrange(0,new_state_length-1):
        # use trapezoidal rule to integrate acceleration
        # this is exact if we assume that acceleration is piecewise linear
        # i.e. first order hold
        v[:,k+1]=v[:,k]+dt_new/2*(a[:,k]+a[:,k+1])
        # use Simpson's rule to integrate velocity
        # this is exact assuming acceleration is piecewise linear, making
        # velocity piecewise quadratic
        # i.e. 2nd order hold
        a_halfk = 0.5*(a[:,k]+a[:,k+1])
        v_halfk=v[:,k]+dt_new/4*(a[:,k]+a_halfk)
        p[:,k+1]=p[:,k]+dt_new/6*(v[:,k]+4*v_halfk+v[:,k+1])

    # interpolate attitude
    interp_att = xyzw_interp(interp_time,original_pose.time,original_pose.attitude)

    # find numerical angular velocity

    # find qdot
    numerical_qdot = xyzw_five_point_stencil(interp_att,dt_interp)
    numerical_w = XYZArray(numerical_qdot.length())
    # find w given qdot and q
    for k in xrange(0,numerical_qdot.length()):
        # create quaternion from q data
        q_k = Quat(interp_att[:,k+2],"xyzw")
        qdot_k = numerical_qdot[:,k:k+1]
        numerical_w[:,k:k+1] = np.dot(2*q_k.Xi().T,qdot_k)
    # interpolate w to get more "resolution"
    numerical_w_times = interp_time[2:-2]
    new_state.angular_rate = xyz_interp(new_state.time,numerical_w_times,numerical_w)

    # integrate w to get new attitude

    # create views for notational convenience
    q=new_state.attitude
    w=new_state.angular_rate
    # set initial condition
    q[:,0] = interp_att[:,2]
    for k in xrange(0,new_state_length-1):
        # create AngularRate objects so that we can get the Omega matrix
        w_k = AngularRate(w[:,k])
        w_kplus = AngularRate(w[:,k+1])
        w_bar = 0.5*(w_k+w_kplus)
        q_k = q[:,k:k+1]
        # integrate w using 1st order hold
        # this is exact assuming w is piecewise linear
        q[:,k+1:k+2] = dot(expm(0.5*w_bar.Omega()*dt_new)+1/48.*(dot(w_kplus.Omega(),w_k.Omega())-dot(w_k.Omega(),w_kplus.Omega()))*dt_new**2,q_k)

    return new_state
