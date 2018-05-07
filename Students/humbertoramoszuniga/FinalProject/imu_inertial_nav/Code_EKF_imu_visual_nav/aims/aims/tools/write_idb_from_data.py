
bag_directory = "/data/aims"
bag_name = "orange_2.bag"
#bag_name = "space_lab.bag"
imu_topic_str = "/vectornav/imu"
camera_topic_str = "camera/image_raw"
mocap_topic_str = "/vicon/LENS12NED/LENS_NED"
range_topic_str = "/mavros/px4flow/ground_distance"


#database_path = "Dropbox/aims_db/space_lab.idb"
database_path = "Dropbox/aims_db/orange_2.idb"

import aims
import numpy as np
from aims.attkins import Quat

tf=118

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
idb.reference.pose = aims.tools.pose_from_bag(bag_directory,bag_name,mocap_topic_str,save_time)

R=(0.01/3.)
rate=10
idb.gps = aims.tools.gps_from_bag(bag_directory,bag_name,mocap_topic_str,save_time,R,rate)

idb.write_to_file()

print idb.reference.pose.position[:,0].tolist()
print idb.reference.pose.attitude[:,0].tolist()