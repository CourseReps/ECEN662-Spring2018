"""
Order should be x-down, y-down, z-down
"""

import numpy as np
from numpy import dot
from numpy.linalg import multi_dot, inv,norm
import os

import matplotlib.pylab as plt

from scipy.optimize import leastsq,minimize

import rosbag

from multiplot2d import MultiPlotter
from aims.attkins import Quat, skew_symmetric

np.set_printoptions(precision=4,suppress=True,linewidth=250)


data_dir="/data/vn100/mag"
bag_names = ("x_down.bag", "y_down.bag","z_down.bag")
imu_topic = "/vectornav"

#g = 

class RateHistory(object):
    
    def __init__(self,t,a,w,m):
        self.t = t
        self.s = a
        self.w = w
        self.m = m

def history_from_bag(bag_name,t_max=None):

    bag = rosbag.Bag(bag_name)
    
    # inform user
    print "Reading " + bag_name
    
    # preallocate data array
    imu_topic_str = imu_topic +"/imu"
    num_msgs = bag.get_message_count(imu_topic_str)
    time = np.zeros(num_msgs)
    w = np.zeros((num_msgs,3))
    s = np.zeros((num_msgs,3))
    m = np.zeros((num_msgs,3))
    
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
       
    mag_topic_str = imu_topic +"/mag"
    # loop through received messages and store data in data array
    i=0
    for topic, msg, t in bag.read_messages(topics=mag_topic_str):
        t=msg.header.stamp.to_sec()
        # use first time as reference time
        if i==0:
            t0=t
        m_msg = msg.magnetic_field
        time[i] = t-t0
        m[i,:] = np.array([m_msg.x,m_msg.y,m_msg.z])
        i+=1
        
        if t_max != None and (t-t0)>= t_max:
            break
        
    return RateHistory(time[:i],s[:i,:],w[:i,:],m[:i,:])

# gravity in world NED frame
g_w = np.array([ [0.,0.,9.79343] ]).T

m_w=np.zeros((3,1))
m_w[0,0]=24087.3	
m_w[1,0]=1321.6
m_w[2,0]=41016.4
m_w/=1e4

def y_hat(q,accel_bias,mag_bias):
    # attitude  
    q2=np.sum(dot(q,q))
    k=1./(1.+q2)
    q_x = skew_symmetric(q)
    C = np.eye(3)+2.*k*(-q_x+dot(q_x,q_x))
    
    s_hat = -dot(C,g_w)+accel_bias    
    m_hat = dot(C,m_w)+mag_bias
    
    return (s_hat,m_hat)

def residual(x,data1,data2,data3):
    data_list = (data1,data2,data3)
    q1=x[:3]
    q2=x[3:6]
    q3=x[6:9]
    accel_bias=np.array([x[9:12]]).T
    mag_bias=np.array([x[12:15]]).T
    
    q_list = (q1,q2,q3)
    
    num_measurement_list = [3*np.shape(data.s)[0] for data in data_list]
    total_num_measurements = sum(num_measurement_list*2)
    residual = np.zeros(total_num_measurements)
    
    i=0
    for q,data,data_length in zip(q_list,data_list,num_measurement_list):
  
        s_hat,m_hat = y_hat(q,accel_bias,mag_bias)
        # acceleration residual 
        residual[i:i+data_length] = (data.s-s_hat.flatten()).flatten()
        i+=data_length
        
        # mag residual
        residual[i:i+data_length]  = (data.m-m_hat.flatten()).flatten()
        i+=data_length
    return residual

def crp_vector_jacobian(q,v):

    J=np.zeros((3,3))
    
    q=q.flatten()
    q2=np.sum(dot(q,q))
    k=1./(1.+q2)
    q_x = skew_symmetric(q)
    v_x = skew_symmetric(v)
    q=np.array([ q ]).T
  
    
    h=-dot(q_x,v)
    g=multi_dot([q_x,q_x,v])
    J1=-2*k**2*(h+g)
    J2=2*q.T


    J3=2*k*np.eye(3)
    J4=v_x
    
    J5=J3
    J6=-dot(q_x,v_x)+skew_symmetric(dot(v_x,q))
   
    J = dot(J1,J2)+dot(J3,J4)+dot(J5,J6)


    return J

def jacobian(x,data1,data2,data3):
    q1=x[:3]
    q2=x[3:6]
    q3=x[6:9]
    accel_bias=np.array([x[9:12]]).T
    mag_bias=np.array([x[12:15]]).T
    
    data_list = (data1,data2,data3)
    
    q_list = (q1,q2,q3)
    
    num_measurement_list = [3*np.shape(data.s)[0] for data in data_list]
    total_num_measurements = sum(num_measurement_list*2)
    H = np.zeros((total_num_measurements,15))
    
  
    i=0
    row=0
    for q,data_length in zip(q_list,num_measurement_list):
        col_slice = slice(i*3,i*3+3)
        H_copy = np.zeros((6,15))
        H_copy[:,-6:] = np.eye(6)
        
        H_copy[:3,col_slice] = -crp_vector_jacobian(q,g_w)   
        H_copy[3:6,col_slice] = crp_vector_jacobian(q,m_w)  
#        print data_length*2
        H[row:row+data_length*2,:]=np.tile(H_copy,(data_length/3,1))
        
        i+=1      
        row+=data_length*2
    return H

    
        

plt.close("all")    
plt.style.use("dwplot")        
def plot_results(x,data_list,size_inches):
    # create a plot for acceleration 
    s_plotter = MultiPlotter((3,3),size_inches=size_inches,name="Accelerometer Noise")
    m_plotter = MultiPlotter((3,3),size_inches=size_inches,name="Mag Noise")
    
    plotter_list = (s_plotter,m_plotter)
    for plotter in plotter_list:
        plotter.set_axis_titles("all","Time[s]","")
        plotter.set_plot_titles(range(3),"x")
        plotter.set_plot_titles(range(3,6),"y")
        plotter.set_plot_titles(range(6,9),"z")
        plotter.add_grid("all")
    
    q1=x[:3]
    q2=x[3:6]
    q3=x[6:9]
    accel_bias=np.array([x[9:12]]).T
    mag_bias=np.array([x[12:15]]).T
    
    q_list = (q1,q2,q3)
    
    i=0
    for q,data in zip(q_list,data_list):
        s_hat,m_hat = y_hat(q,accel_bias,mag_bias)
        plot_list = (i,3+i,6+i)
        
        s_noise = data.s - s_hat.flatten()
        s_plotter.add_data(plot_list,data.t,s_noise)
        
        m_noise = data.m - m_hat.flatten()
        m_plotter.add_data(plot_list,data.t,m_noise)
        i+=1
        
    for plotter in plotter_list:
        plotter.display()

# change to bag directory
os.chdir(os.environ['HOME']+data_dir)

x_down = history_from_bag(bag_names[0])
y_down = history_from_bag(bag_names[1])
z_down = history_from_bag(bag_names[2])

data = (x_down,y_down,z_down)

def AxisAngle2CRP(angle,axis):
    return np.tan(angle/2.)*axis

# x-down initial attitude guess
x1=AxisAngle2CRP(-np.pi/2.,np.array([0,1,0]))
# y-down initial attitude guess
x2=AxisAngle2CRP(np.pi/2.,np.array([1,0,0.7776]))
# z-down initial attitude guess
x3=AxisAngle2CRP(0.,np.array([0,0,1]))

bias=np.zeros(6)

x=np.concatenate([x1,x2,x3,bias])

tol = 1e-3
newton_iteration = 0
delta_optimize_variables = 100
while delta_optimize_variables >= tol:
    res = np.array([residual(x,*data)]).T
    J = jacobian(x,*data)
    dx=multi_dot([inv(dot(J.T,J)),J.T,res])
    delta_optimize_variables = norm(dx) 
    x+=dx.flatten()
    

#print residual(x0,data)

#jacobian(x0,data)

#ret = minimize(residual,x0,data,jac=jacobian)
#print ret




plot_results(x,data,(15,8))

m_avg = np.mean(x_down.m,axis=0)
print np.pi/2-np.arctan2(m_avg[0],m_avg[1])
m_avg = np.mean(y_down.m,axis=0)
print np.pi/2-np.arctan2(m_avg[0],m_avg[1])
m_avg = np.mean(z_down.m,axis=0)
print np.pi/2-np.arctan2(m_avg[0],m_avg[1])