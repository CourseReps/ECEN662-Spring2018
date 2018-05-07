
bag_directory = "/data/aims"
bag_name = "large_depth_1.bag"
mocap_topic_name = "/LENS/pose"

database_path = "Dropbox/aims_db/sim.idb"
from aims import *

tf=90

t0=tools.get_initial_time(bag_directory,bag_name,mocap_topic_name)
tf+=t0
save_time=(t0,tf)

# get recorded pose
pose = tools.pose_from_bag(bag_directory,bag_name,mocap_topic_name,save_time)

# get approximate kinematic history of pose
kinematics = sim.kinematics_from_pose(pose)
kinematics.position.y -= 4
#kinematics.position.x += 2

my_database = InputDatabase(database_path)
my_database.reference.kinematics = kinematics
my_database.reference.features_xyz = sim.room_features(40)

system_params = SystemParams()
system_params.imu_params.rate = 200
imu_sensor, feature_sensor, range_sensor = sim.system_sensor_sim(kinematics,my_database.reference.features_xyz,system_params)

my_database.imu = imu_sensor
my_database.rangefinder = range_sensor
my_database.feature_detector = feature_sensor
my_database.reference.system_params = system_params
my_database.system_params = system_params

my_database.write_to_file()

