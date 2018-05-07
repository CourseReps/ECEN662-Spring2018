import numpy as np
from numpy import dot
from numpy.linalg import multi_dot, inv
import os

import matplotlib.pylab as plt

import rosbag

from multiplot2d import MultiPlotter
from aims.attkins import Quat

np.set_printoptions(precision=4,suppress=True,linewidth=250)


data_dir="/data/vn100/mag"
bag_names = ("x_down.bag", "y_down.bag","z_down.bag")
imu_topic_str = "/vectornav/imu"

g = 9.79343

class RateHistory(object):
    
    def __init__(self,t,a,w):
        self.t = t
        self.s = a
        self.w = w

def history_from_bag(bag_name,t_max=None):

    bag = rosbag.Bag(bag_name)
    
    # inform user
    print "Reading " + bag_name
    
    # preallocate data array
    num_msgs = bag.get_message_count(imu_topic_str)
    time = np.zeros(num_msgs)
    w = np.zeros((num_msgs,3))
    s = np.zeros((num_msgs,3))
    
    # loop through received messages and store data in data array
    i=0
    for topic, msg, t in bag.read_messages(topics=imu_topic_str):
        t=msg.header.stamp.to_sec()
        # use first time as reference time
        if i==0:
            t0=t
        w_msg = msg.angular_velocity
        s_msg = msg.linear_acceleration
        time[i] = t-t0
        w[i,:] = np.array([w_msg.x,w_msg.y,w_msg.z])
        s[i,:] = np.array([s_msg.x,s_msg.y,s_msg.z])
        i+=1
        
        if t_max != None and (t-t0)>= t_max:
            break
        
    return RateHistory(time[:i],s[:i,:],w[:i,:])

# change to bag directory
os.chdir(os.environ['HOME']+data_dir)

x_down = history_from_bag(bag_names[0])
y_down = history_from_bag(bag_names[1])
z_down = history_from_bag(bag_names[2])

s_tilde_list = (x_down.s, y_down.s, z_down.s)
w_tilde_list = (x_down.w, y_down.w, z_down.w)
time_list = (x_down.t, y_down.t, z_down.t)
num_measurement_list = [np.shape(s)[0] for s in s_tilde_list]

###############################################################################
# PROCESS ACCELERATION MEASUREMENTS
###############################################################################

# define measurement sensitivity matrices
zero_32 = np.zeros((3,2))
I3 = np.eye(3)
alpha1 = np.array([ [0,0], [0, g], [-g,0] ])
H1=np.concatenate([alpha1, zero_32, zero_32, I3],axis=1)
alpha2 = np.array([ [0,-g], [0, 0], [g,0] ])
H2=np.concatenate([zero_32, alpha2, zero_32, I3],axis=1)
alpha3 = np.array([ [0,g], [-g, 0], [0,0] ])
H3=np.concatenate([zero_32, zero_32, alpha3, I3],axis=1)
H_list = (H1,H2,H3)

# define the acceleration due to gravity for each frame 
ag_1 = np.array([ [g,0,0]] ).T
ag_2 = np.array([ [0,g,0]] ).T
ag_3 = np.array([ [0,0,g]] ).T
ag_list = (ag_1,ag_2,ag_3)

# preallocate combined senstivity matrix and measurement vector
total_num_measurements = sum(num_measurement_list)
H = np.zeros((total_num_measurements*3,9))
z = np.zeros((total_num_measurements*3,1))

# populate combined senstivity matrix and measurement vector
start_row = 0
for s_tilde, ag_i, H_i, n_i in zip(s_tilde_list,ag_list,H_list,num_measurement_list):
    row_slice = slice(start_row,start_row+n_i*3)
    H[row_slice,:] = np.tile(H_i, (n_i,1))
    
    z[row_slice,0] = (s_tilde + ag_i.flatten()).flatten()
    
    start_row += n_i*3

# least squares estimate
x_hat = multi_dot([inv(np.dot(H.T,H)),H.T,z])
bias_accel=x_hat[-3:,0]
angles = x_hat[0:6].flatten()

# convert the "small angle" estimates into quaternion components
q1_vec = np.array([0., angles[0], angles[1],2])*0.5
q2_vec = np.array([angles[2], 0., angles[3],2])*0.5
q3_vec = np.array([angles[4], angles[5], 0.,2])*0.5
q_vec_list = (q1_vec,q2_vec,q3_vec)

# create a plot for acceleration 
plt.close("all")    
plt.style.use("dwplot")
a_plotter = MultiPlotter((3,3),size_inches=(12,5),name="Accelerometer Noise")
a_plotter.set_axis_titles("all","Time[s]","")
a_plotter.set_plot_titles(range(3),"x")
a_plotter.set_plot_titles(range(3,6),"y")
a_plotter.set_plot_titles(range(6,9),"z")

# preallocate a vector for the estimated noise
noise_accel = np.zeros((total_num_measurements,3))

# use the estimated parameters to estimate the noise and plot it
start_row = 0
for i in range(3):
    plot_list = (i,3+i,6+i)
    time = time_list[i]
    s_tilde = s_tilde_list[i]
    ag = ag_list[i]
    q_vec = q_vec_list[i]
    q=Quat(q_vec,order="xyzw")
    q.normalize()
    noise_i = s_tilde - bias_accel + dot(q.asDCM(),ag.flatten())
    a_plotter.add_data(plot_list,time,noise_i)
    
    n_i = num_measurement_list[i]
    row_slice = slice(start_row,start_row+n_i)
    noise_accel[row_slice,:] = noise_i
    start_row += n_i

# find the noise standard deviation
noise_accel_std = np.std(noise_accel,axis=0)

# plot 3-sigma bounds on noise
for i in range(3):
    plot_list = (i,3+i,6+i)
    time = time_list[i]
    t0=time[0]
    tf=time[-1]
    t_bound = np.array([t0,tf])
    three_sigma = np.tile(noise_accel_std,(2,1))*3
    bound_style = dict(color="k",ls="--")
    a_plotter.add_data(plot_list,t_bound,three_sigma,line_styles=bound_style)
    a_plotter.add_data(plot_list,t_bound,-three_sigma,line_styles=bound_style)

a_plotter.display()

###############################################################################
# PROCESS ANGULAR RATE MEASUREMENTS
###############################################################################

all_gyro_measurements = np.concatenate(w_tilde_list)
bias_gyro = np.mean(all_gyro_measurements,axis=0)
noise_gyro_std = np.std(all_gyro_measurements,axis=0)

# create a plot for gyro 
w_plotter = MultiPlotter((3,3),size_inches=(12,5),name="Gyroscope Noise")
w_plotter.set_axis_titles("all","Time[s]","")
w_plotter.set_plot_titles(range(3),"x")
w_plotter.set_plot_titles(range(3,6),"y")
w_plotter.set_plot_titles(range(6,9),"z")
w_plotter.display()

for i in range(3):
    plot_list = (i,3+i,6+i)
    time = time_list[i]
    w_tilde = w_tilde_list[i]
    noise_i = w_tilde-bias_gyro
    w_plotter.add_data(plot_list,time,noise_i)
    
    t0=time[0]
    tf=time[-1]
    t_bound = np.array([t0,tf])
    three_sigma = np.tile(noise_gyro_std,(2,1))*3
    bound_style = dict(color="k",ls="--")
    w_plotter.add_data(plot_list,t_bound,three_sigma,line_styles=bound_style)
    w_plotter.add_data(plot_list,t_bound,-three_sigma,line_styles=bound_style)
    
print ("\nGyro Bias: \n[%.6f, %.6f, %.6f]" %(bias_gyro[0], bias_gyro[1], bias_gyro[2]) )
print ("\nGyro Noise Std: \n[%.6f, %.6f, %.6f]" %(noise_gyro_std[0], noise_gyro_std[1], noise_gyro_std[2]) )

print ("\nAccel Bias: \n[%.6f, %.6f, %.6f]" %(bias_accel[0], bias_accel[1], bias_accel[2]) )
print ("\nAccel Noise Std: \n[%.6f, %.6f, %.6f]" %(noise_accel_std[0], noise_accel_std[1], noise_accel_std[2]) )