
bag_directory = "/data/aims"
bag_name = "scan_room_3.bag"
mocap_topic_name = "/vicon/LENS12NED/LENS_NED"

database_paths = ("Dropbox/aims_db/scan_room.idb", "Dropbox/aims_db/scan_room_up_down.idb")

from aims import *

from insert_up_down import add_up_down

tf = 75

t0=tools.get_initial_time(bag_directory,bag_name,mocap_topic_name)
tf+=t0
save_time=(t0,tf)

# get recorded pose
pose_default = tools.pose_from_bag(bag_directory,bag_name,mocap_topic_name,save_time)
#corner_times = [28., 45., 60., 71.]
#corner_times = [23, 33, 46., 56.5]
corner_times = [23+2, 33+2.5, 46.+2, 56.5+2]
pose_up_down = add_up_down(pose_default,corner_times,7.,1.5)

poses = (pose_default, pose_up_down)
features = sim.room_features(90,lengths=(7,7,3))

system_params = SystemParams()
system_params.imu_params.rate = 200

for pose,path in zip(poses,database_paths):

    # get approximate kinematic history of pose    
    kinematics = sim.kinematics_from_pose(pose)
    
    my_database = InputDatabase(path)
    my_database.reference.kinematics = kinematics
    my_database.reference.features_xyz = features

    imu_sensor, feature_sensor, range_sensor = sim.system_sensor_sim(kinematics,features,system_params)
    
    my_database.imu = imu_sensor
    my_database.rangefinder = range_sensor
    my_database.feature_detector = feature_sensor
    my_database.reference.system_params = system_params
    my_database.system_params = system_params
    
    my_database.write_to_file()

print my_database.reference.kinematics.position[:,0].tolist()
print my_database.reference.kinematics.attitude[:,0].tolist()

