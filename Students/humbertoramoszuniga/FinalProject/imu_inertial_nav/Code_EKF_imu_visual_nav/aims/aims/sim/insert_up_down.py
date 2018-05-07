
import numpy as np
from aims import *
from copy import deepcopy

def add_up_down(old_pose,times,maneuver_time,height):
    print "Inserting some up and down motion"

    pose = deepcopy(old_pose)

    for i, time in enumerate(times):
        # add_time = time + maneuver_time*i
        # dt = 0.01

        # timestep of interest
        k1 = np.abs(pose.time-time).argmin()
        k2 = np.abs(pose.time-time-maneuver_time).argmin()
        # print k
        # t_k = pose.time[k]
        # k_slice = slice(k,k+1)
        # position at this time
        # p_k = pose.position[:,k_slice]
        # q_k = pose.attitude[:,k_slice]

        # time_to_insert = np.arange(t_k,t_k+maneuver_time,dt)
        # delta_t = np.zeros_like(pose.time)
        # delta_t[k:] = maneuver_time
        # print delta_t[k:]
        # pose.time += delta_t
        # # pose.time = np.insert(pose.time,k,time_to_insert)
        # pose.time = np.concatenate( (pose.time[:k], time_to_insert, pose.time[k:]) )

        # num_vals_insert = time_to_insert.shape[0]

        # q_to_insert = np.tile(q_k,(1,num_vals_insert))
        # pose.attitude = np.hstack( (pose.attitude[:,:k], q_to_insert,pose.attitude[:,k:]  ) )
        # pose.attitude = pose.attitude.view(type=XYZWArray)

        # delta_p = np.zeros( (3,num_vals_insert) )
        delta_p = np.zeros_like(pose.position[:,k1:k2])
        p_domain = np.linspace(0,maneuver_time,delta_p.shape[1])
        delta_p[2,:] = np.exp(-(p_domain-maneuver_time/2)**2)*height

        pose.position[:,k1:k2] += delta_p


        # p_to_insert = np.tile(p_k,(1,num_vals_insert)) + delta_p

        # pose.position = np.hstack( (pose.position[:,:k], p_to_insert,pose.position[:,k:]  ) )
        # pose.position = pose.position.view(type=XYZArray)

    return pose
