
bag_directory = "/data/aims"
bag_name = "medium_depth.bag"
bag_name = "large_depth_2.bag"
bag_name = "general_motion_mixed_depth.bag"
imu_topic_str = "/vectornav/imu"
camera_topic_str = "/pointgrey/image_raw"
#camera_topic_str = "image_raw_throttle"
mocap_topic_str = "/LENS/pose"
range_topic_str = "/mavros/distance_sensor/hrlv_ez4_pub"
#range_topic_str = "/mavros/px4flow/ground_distance"

database_path = "Dropbox/aims_db/medium_depth.idb"
database_path = "Dropbox/aims_db/general_motion_mixed_depth.idb"

import aims
import numpy as np
from aims.attkins import Quat

tf=80

t0=aims.tools.get_initial_time(bag_directory,bag_name,imu_topic_str)
tf+=t0
save_time=(t0,tf)

idb = aims.InputDatabase(database_path)

# get imu data 
idb.imu = aims.tools.imu_from_bag(bag_directory,bag_name,imu_topic_str,save_time) 

# create camera
idb.camera = aims.Camera(bag_directory,bag_name,camera_topic_str,t0)

# get range data
idb.rangefinder = aims.tools.rangefinder_from_bag(bag_directory,bag_name,range_topic_str,save_time)

# get motion capture pose for comparision
q_bi = Quat.simple_rotation(np.pi,"x")*Quat.simple_rotation(np.pi/2,"z")
q_wa = Quat.simple_rotation(-np.deg2rad(180),"z")
idb.reference.pose = aims.tools.pose_from_bag(bag_directory,bag_name,mocap_topic_str,save_time,q_bi,q_wa)

idb.write_to_file()



