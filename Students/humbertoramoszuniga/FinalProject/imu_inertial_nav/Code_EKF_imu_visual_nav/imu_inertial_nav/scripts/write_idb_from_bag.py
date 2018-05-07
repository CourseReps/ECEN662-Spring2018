bag_directory = "/data"
# bag_directory = "/Dropbox/aims_db/humberto/"
bag_name = "data_3.bag"
# bag_name = "data_and_camera_2.bag"
database_path = "data/data_3.idb"
# database_path = "data/data_and_camera_2.idb"

imu_topic_str = "/vectornav/imu"
camera_topic_str = "/image_raw"
mocap_imu_topic_str = "/vicon/LENS12NED/LENS_NED"
mocap_target_topic_str = "/vicon/aruco_target/board"
corner_topic_str = "/aruco/corners"

import rpmsat.tools as tools
from rpmsat import KinematicArray, Camera
import numpy as np
from quatematics import Quat

from project_database import InputDatabase
from useful_functions import viconToTargetMap

tf = 118

t0 = tools.get_initial_time(bag_directory, bag_name, corner_topic_str)
tf += t0
save_time = (t0, tf)

idb = InputDatabase(database_path)

# get imu data 
idb.imu = tools.imu_from_bag(bag_directory, bag_name, imu_topic_str, save_time)

# get feature detector
idb.feature_detector = tools.feature_detector_from_bag(bag_directory, bag_name, corner_topic_str, save_time)

# get motion capture pose for comparision
pose_imu_vicon = tools.pose_from_bag(bag_directory, bag_name, mocap_imu_topic_str, save_time)

pose_target_vicon = tools.pose_from_bag(bag_directory, bag_name, mocap_target_topic_str, save_time)

# position of target in vicon frame
# take average of (assumed static) vicon measurements
p_vicon_t = np.mean(pose_target_vicon.position, axis=1).reshape(3, 1)

# attitude of target in vicon frame
q_vicon_t = np.mean(pose_target_vicon.attitude, axis=1)
# q_vicon_t.normalize()

len_pose = pose_imu_vicon.position.length()
pose_imu_target = KinematicArray(len_pose)
pose_imu_target.time = pose_imu_vicon.time

for k in xrange(len_pose):
    k_slice = slice(k, k + 1)
    q_vicon_I = pose_imu_vicon.attitude[:, k_slice]
    p_vicon_I = pose_imu_vicon.position[:, k_slice]
    q_t_I, p_t_I = viconToTargetMap(q_vicon_I, q_vicon_t, p_vicon_I, p_vicon_t)
    pose_imu_target.position[:, k_slice] = p_t_I
    pose_imu_target.attitude[:, k_slice] = q_t_I.asColVector()

idb.reference.kinematics = pose_imu_target
idb.target_attitude = Quat(q_vicon_t)

# create camera
idb.camera = Camera(bag_directory, bag_name, camera_topic_str, t0)

# save to file
idb.write_to_file()

w_data = np.zeros((3, len(idb.imu.data)))
s_data = np.zeros((3, len(idb.imu.data)))
for k, msg in enumerate(idb.imu.data):
    w_data[:, k] = msg.angular_rate.flatten()
    s_data[:, k] = msg.acceleration.flatten()

n_sec = 5
rate = 200
k_max = n_sec * rate
w_bias_est = np.mean(w_data[:, :k_max], axis=1)

#
# print idb.reference.pose.position[:,0].tolist()
# print idb.reference.pose.attitude[:,0].tolist()
