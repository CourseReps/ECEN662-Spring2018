from scipy.interpolate import interp1d
from aims.attkins import Quat
from aims.core import EstimatedState
from aims import *
import numpy as np
from numpy import arcsin, arccos, cos 

#length=np.sum( np.sqrt( np.sum( np.diff(data_set,1,1)**2 ,2)) ,1)


def quat_2_euler_321(quat_history):
    euler_321 = np.zeros((3,quat_history.shape[1]))
    
    for i in xrange(quat_history.shape[1]):
        quat = Quat(quat_history[:,i:i+1])
        quat.normalize()
        C = quat.asDCM()
        C02 = C[0,2]
        C12 = C[1,2]
        C00 = C[0,0]
        a2 = arcsin(-C02)
        a1 = arccos(C00/cos(a2))
        a3 = arcsin(C12/cos(a2))
        euler_321[:,i] = [a1,a2,a3]
        
    return euler_321
    

def interpolate_estimates(est_history,ref_pose,components):
    interp_est = EstimatedState(ref_pose.length())
    interp_est.time = ref_pose.time
    
    for comp in components:
        original_data = getattr(est_history,comp)
        original_time = est_history.time
        interp_fun = interp1d(original_time,original_data,axis=1,bounds_error=False)
        interp_vals = interp_fun(ref_pose.time)
        setattr(interp_est,comp,interp_vals)
        
    return interp_est

def interpolate_three_sigma(three_sigma,new_time):
    new_three_sigma = ThreeSigma(new_time.shape[0])
    new_three_sigma.time = new_time
    interp_fun = interp1d(three_sigma.time,three_sigma.imu_state,axis=1)
    new_three_sigma.imu_state = interp_fun(new_time)
    return new_three_sigma

def additive_error(est_history,ref_pose,components):
    errors = EstimatedState(ref_pose.length())
    errors.time = ref_pose.time
    
    for comp in components:
        ref_vals = getattr(ref_pose,comp)
        est_vals = getattr(est_history,comp)
        setattr(errors,comp,ref_vals-est_vals)
    return errors