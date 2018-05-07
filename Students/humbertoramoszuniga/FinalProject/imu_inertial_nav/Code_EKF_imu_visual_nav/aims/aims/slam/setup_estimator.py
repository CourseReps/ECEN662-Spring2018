import os
import math
import copy

import numpy as np
import yaml

from aims import *
from aims.slam import EKFSLAM,InitialCalibrator
from aims.attkins import Quat

def setup_estimator(yaml_path,save_dt):
    
    with open(os.environ['HOME']+yaml_path, 'r') as yaml_stream:
        try:
            yaml_dict = yaml.load(yaml_stream)
        except yaml.YAMLError as exc:
            print ("Error opening YAML.")   
    
    idb_path = yaml_dict["idb_path"]
    
    # open input database
    # alias names for the sensors
    idb = database_from_file(idb_path)
    
    t0=yaml_dict["time_start"]
    # final time
    tf = yaml_dict["time_end"]
    print tf
    
       
    # create clock, which handles the simulation "time"
    # each sensor gets a reference to the clock
    # the sensors will use the clock to "know" when new measurements are 
    # available
    dt=yaml_dict["dt"]
    clock = Clock(t0,tf,dt)
    if idb.feature_detector != None:
        idb.feature_detector.add_clock(clock)
    if idb.camera != None:
        idb.camera.add_clock(clock)
    if idb.gps != None:
        idb.gps.add_clock(clock)
    idb.rangefinder.add_clock(clock)
    idb.imu.add_clock(clock)
    
    # set initial conditions 
    x0_dict = yaml_dict["x0"]
    state_0=slam.EstimatedState()
    state_0.p_i_w = np.array([x0_dict["p_i_w"] ]).T
    state_0.v_i_w = np.array([x0_dict["v_i_w"] ]).T
    state_0.q_wi = Quat(x0_dict["q_wi"], order="xyzw")
    state_0.q_wi.normalize()    
    state_0.b_g = np.array([x0_dict["gyro_bias"] ]).T
    state_0.b_a = np.array([x0_dict["accel_bias"] ]).T
    
    # set initial covariance 
    three_sigma_dict = yaml_dict["x0_3_sigma"]
    sigma_theta_wi = np.deg2rad([three_sigma_dict["theta_wi"]]*3)
    sigma_p_i_w = [three_sigma_dict["p_i_w"]]*3
    sigma_v_i_w = [three_sigma_dict["v_i_w"]]*3
    sigma_gyro_bias= [three_sigma_dict["gyro_bias"]]*3
    sigma_accel_bias = [three_sigma_dict["accel_bias"]]*3
    
    P0_diag = np.power(np.concatenate(
        [sigma_theta_wi, sigma_p_i_w, sigma_v_i_w, sigma_gyro_bias,
         sigma_accel_bias])/3.,2)
    P0 = np.diag(P0_diag)
#    print np.diag(P0)
#    dd
    
    ekf_params = slam.EKFParams()    
    # set camera parameters 
    camera_dict = yaml_dict["camera"]
    camera_params = ekf_params.system_params.camera_params  
    camera_params.cx = camera_dict["cx"]
    camera_params.cy = camera_dict["cy"]
    camera_params.fx = camera_dict["fx"]
    camera_params.fy = camera_dict["fy"]
    camera_params.pixel_noise = camera_dict["pixel_noise"]
    camera_params.distortion = np.array(camera_dict["distortion"])
    
    # set imu parameters 
    imu_dict = yaml_dict["imu"]
    imu_params = ekf_params.system_params.imu_params 
    imu_params.gyro_noise = imu_dict["gyro_noise"]
    imu_params.gyro_walk = imu_dict["gyro_walk"]
    imu_params.accel_noise = imu_dict["accel_noise"]
    imu_params.accel_walk = imu_dict["accel_walk"]    
    
    # set intersensor parameters 
    intersensor_dict = yaml_dict["intersensor"]
    intersensor_params = ekf_params.system_params.intersensor_params
    intersensor_params.p_c_i = np.array([intersensor_dict["p_c_i"] ]).T
    intersensor_params.p_r_i = np.array([intersensor_dict["p_r_i"] ]).T
    intersensor_params.q_ic = Quat(intersensor_dict["q_ic"], order="xyzw")
    intersensor_params.q_ic.normalize()
    intersensor_params.q_ir = Quat(intersensor_dict["q_ir"], order="xyzw")
    intersensor_params.q_ir.normalize()
    
    # set feature parameters 
    intersensor_dict = yaml_dict["features"]
    feature_params = ekf_params.system_params.feature_params   
    feature_params.min_init_confidence = intersensor_dict["min_init_confidence"]
    feature_params.max_missed_frames = intersensor_dict["max_missed_frames"]
    feature_params.new_keyframe_threshold = intersensor_dict["new_keyframe_threshold"]
    
    ekf_params.gravity = np.array([ yaml_dict["gravity"]  ]).T

    # set range parameters    
    range_dict = yaml_dict["rangefinder"]
    range_params = ekf_params.system_params.range_params 
    range_params.noise = range_dict["range_noise"]
    
    odb = OutputDatabase(yaml_dict["odb_path"])  
    odb.idb = idb
    
    length = math.ceil(tf-t0)/save_dt+1
    state_history = EstimatedState(length)
    three_sigma_history = ThreeSigma(length)
    odb.estimate_history = state_history
    odb.three_sigma = three_sigma_history
    odb.feature_count =  FeatureCount(length)    
    
    calibrator = InitialCalibrator(yaml_dict["time_calibrate"],state_0,P0,ekf_params)

    
    return [calibrator,clock,odb] 