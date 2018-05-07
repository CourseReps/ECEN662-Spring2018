
import time
import numpy as np
from aims.slam import EKFSLAM
from aims.viz import ROSViz
import rospy

from setup_estimator import setup_estimator

import numpy as np
from numpy.linalg import eigvals, eig

np.set_printoptions(precision=6,suppress=True,linewidth=350)

yaml_dir = "/catkin_ws/src/aims/aims/config/"

#yaml_name = "scan_room.yaml"
#yaml_name = "sim_space_lab.yaml"
yaml_name = "space_lab.yaml"
#yaml_name = "orange_2.yaml"

save_dt = 0.05
[calibrator,clock,odb] = setup_estimator(yaml_dir+yaml_name,save_dt)
imu=odb.idb.imu
feature_detector=odb.idb.feature_detector
rangefinder=odb.idb.rangefinder
camera = odb.idb.camera
gps = odb.idb.gps

while clock():
    if imu.poll():
        calibrator.register_imu( imu.latest() )
        
    if calibrator.ready():
        calibrator.estimate()
        break

estimator=EKFSLAM(calibrator.x0,calibrator.P0,calibrator.ekf_params,clock)
rospy.init_node('ekf_odom', anonymous=False)

viz = ROSViz(estimator,clock,odb.idb.reference)

# main loop
t_start = time.time()
k=0

last_save_t = 0.

while clock() and not rospy.is_shutdown():

        
    if feature_detector != None and feature_detector.poll():
        estimator.add_measurement(feature_detector.latest())
#        
    if camera != None and camera.poll():
        estimator.add_measurement(camera.latest())
            
    if rangefinder.poll():
        estimator.add_measurement(rangefinder.latest())
    
    if imu.poll():
        estimator.add_measurement(imu.latest())    
    
    if (clock.now()-last_save_t) >= save_dt:
        last_save_t = clock.now()
        odb.estimate_history.attitude[:,k:k+1] = estimator.state.q_wi.asColVector()    
        odb.estimate_history.velocity[:,k:k+1] = estimator.state.v_i_w   
        odb.estimate_history.position[:,k:k+1] = estimator.state.p_i_w   
        odb.estimate_history.time[k] = clock.now() 
        odb.three_sigma.time[k] = clock.now() 
        odb.three_sigma.imu_state[:,k:k+1] = estimator.three_sigma()[:15,:]
#        Pff = estimator.feature_covariance()
        
        k+=1
        
        
        odb.feature_count.features_in_map[k] = estimator.manager.num_features
        odb.feature_count.visible_features[k] = len(estimator.visible_features)
    
    clock.print_throttled(1)    
    
    
    viz.update()

rospy.signal_shutdown("done")

if camera != None:
    camera.close()
t_end = time.time()
print "True Run time: " + str(t_end-t_start)

odb.write_to_file()


