
import numpy as np
import os

from rpmsat import database_from_file, Clock

import matplotlib.pylab as plt
from multiplot2d import MultiPlotter

#######
# INPUT 
#######

# where is the input file?
database_path = "/data/space_lab_sim_no_imu_noise.idb"

# the simulated clock with a finite time step
dt = 0.00001
# when does the clock start?
t0 = 0.# when does the clock end?
tf = 20.

###############
# PROCESS INPUT
###############

# load input file
idb = database_from_file(os.environ['HOME']+database_path)

# create clock, which handles the simulation "time"
# each sensor gets a reference to the clock
# the sensors will use the clock to "know" when new measurements are 
# available
clock = Clock(t0,tf,dt)

# create view of sensors
imu=idb.imu
feature_detector=idb.feature_detector

# add the clock to the sensors
imu.add_clock(clock)
feature_detector.add_clock(clock)

while clock():
    if imu.poll():
        print "clock time: ", clock.now()
        imu_measurement = imu.latest()
        print "IMU message stamp: ", imu_measurement.true_time
        print "w: ", imu_measurement.angular_rate.flatten()
        print "s: ", imu_measurement.acceleration.flatten()
        print "\n"
        
    if feature_detector.poll():
        print "clock time: ", clock.now()
        feature_measurement = feature_detector.latest()
        print "feature message stamp: ", feature_measurement.true_time
        print "%d features detected" % len(feature_measurement.features)

        for feature in feature_measurement.features:
            # first_feature = feature_measurement.features[0]
            print "the global position of the first feature: ", feature.p_f_w.flatten()
            print "the pixel coordinates of the feature: [", feature.u, feature.v, "]"
            print "\n"

ref_time = idb.reference.kinematics.time
ref_position_history = idb.reference.kinematics.position
ref_attitude_history = idb.reference.kinematics.attitude

######
# PLOT
######

# setup plotting
plt.close("all")
# plt.style.use("dwplot")

plotter_list    = []

# create and add data to plotter for position
pos_plotter = MultiPlotter((3,1), (6,6),"Position",dict(sharex=True))
pos_plotter.add_data("all",ref_time,ref_position_history.T,labels="Reference")
pos_plotter.set_axis_titles(2,"Time")
pos_plotter.set_plot_titles("all",["x","y","z"])

plotter_list.append(pos_plotter)

# create and add data to plotter for attitude
att_plotter = MultiPlotter((2,2), (6,5),"Attitude",dict(sharex=True))
att_plotter.add_data("all",ref_time,ref_attitude_history.T,labels="Reference")
att_plotter.set_axis_titles((2,3),"Time")
att_plotter.set_plot_titles("all",["x","y","z","w"])

plotter_list.append(att_plotter)

# apply some settings and display all plotters
for plotter in plotter_list:
    plotter.add_grid("all")
    plotter.add_figure_legend(legend_space_inches=0.15)
    plotter.display(False)
