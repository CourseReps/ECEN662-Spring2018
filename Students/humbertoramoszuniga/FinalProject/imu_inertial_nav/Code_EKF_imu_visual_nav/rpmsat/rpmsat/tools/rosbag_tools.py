
import sys
import rosbag
import os
import numpy as np

from numpy.random import normal

from cv_bridge import CvBridge, CvBridgeError
import cv2
from matplotlib import pyplot as plt

from quatematics import Quat

from rpmsat import IMU, RangeFinder, FeatureDetector, PoseArray, PixelFeature

def get_initial_time(bag_directory,bag_name,topic_str):
    os.chdir(os.environ['HOME']+bag_directory)

    print "Reading " + bag_name
    bag = rosbag.Bag(bag_name)
    topic, msg, t = bag.read_messages(topics=topic_str).next()
    t=msg.header.stamp.to_sec()
    return t

def gps_from_bag(bag_directory,bag_name,mocap_topic_name,save_time,R,rate):
    """
    Get position history from bagged TransformStamped messges.

    """
    t0=save_time[0]
    tf=save_time[1]

    bag,num_msgs = get_bag(bag_directory,bag_name,mocap_topic_name)

    gps_sensor = GPS(num_msgs)

    i=0
    t_last = 0.
    for topic, msg, t in bag.read_messages(topics=mocap_topic_name):
        t=msg.header.stamp.to_sec()
        if t >= t0 and t<=tf and (t-t_last) >= 1./rate:
            t_last = t

            gps_measurement=gps_sensor.data[i]

            p_msg = msg.transform.translation
            q_msg = msg.transform.rotation
            gps_measurement.position[:,0] = [p_msg.x, p_msg.y, p_msg.z]
            gps_measurement.position[:,:] += normal(0,R,size=(3,1))
            gps_measurement.true_time = t-t0
            i+=1

    gps_sensor.data = gps_sensor.data[:i]

    return gps_sensor

def rangefinder_from_bag(bag_directory,bag_name,range_topic_name,save_time):
    t0=save_time[0]
    tf=save_time[1]

    bag,num_msgs = get_bag(bag_directory,bag_name,range_topic_name)

    range_sensor = RangeFinder(num_msgs)

    i=0
    for topic, msg, t in bag.read_messages(topics=range_topic_name):
        t=msg.header.stamp.to_sec()
        if t >= t0 and t<=tf:
            if msg.range > msg.min_range:
                range_measurement = range_sensor.data[i]
                range_measurement.range = msg.range
                range_measurement.true_time = t-t0
                i+=1
    range_sensor.data = range_sensor.data[:i]
    return range_sensor

def imu_from_bag(bag_directory,bag_name,imu_topic_name,save_time):
    t0=save_time[0]
    tf=save_time[1]

    bag,num_msgs = get_bag(bag_directory,bag_name,imu_topic_name)

    imu_sensor = IMU(num_msgs)

    i=0
    for topic, msg, t in bag.read_messages(topics=imu_topic_name):
        t=msg.header.stamp.to_sec()
        if t >= t0 and t<=tf:
            t=msg.header.stamp.to_sec()
            w_msg = msg.angular_velocity
            s_msg = msg.linear_acceleration

            imu_measurement = imu_sensor.data[i]
            imu_measurement.true_time = t-t0
            imu_measurement.seq = i
            imu_measurement.angular_rate[:,0]  = np.array([w_msg.x,w_msg.y,w_msg.z])
            imu_measurement.acceleration[:,0] = np.array([s_msg.x,s_msg.y,s_msg.z])
            i+=1

    imu_sensor.data = imu_sensor.data[:i]
    return imu_sensor

def pose_from_bag(bag_directory,bag_name,mocap_topic_name,save_time):
    """
    Get pose history from bagged TransformStamped messges.

    """
    t0=save_time[0]
    tf=save_time[1]

    bag,num_msgs = get_bag(bag_directory,bag_name,mocap_topic_name)

    pose = PoseArray(num_msgs)

    k=0
    for topic, msg, t in bag.read_messages(topics=mocap_topic_name):
        t=msg.header.stamp.to_sec()
        if t >= t0 and t<=tf:

            p_msg = msg.transform.translation
            q_msg = msg.transform.rotation
            pose.position[:,k] = [p_msg.x, p_msg.y, p_msg.z]

            q_wb_vec = [q_msg.x, q_msg.y, q_msg.z, q_msg.w]

            q_wb = Quat(q_wb_vec)
            q_wb.normalize()

            pose.attitude[:,k:k+1] = q_wb.asColVector()
            pose.time[k] = t-t0
            k+=1

    pose.attitude=pose.attitude[:,:k]
    pose.position=pose.position[:,:k]
    pose.time=pose.time[:k]


    return pose

def feature_detector_from_bag(bag_directory,bag_name,corner_topic_str,save_time):
    """
    Get location of Aruco corners from bag..

    """
    t0=save_time[0]
    tf=save_time[1]

    bag,num_msgs = get_bag(bag_directory,bag_name,corner_topic_str)

    detector = FeatureDetector(num_msgs)
    i=0
    for topic, msg, t in bag.read_messages(topics=corner_topic_str):
        t=msg.header.stamp.to_sec()

        if t >= t0 and t<=tf:

            feature_measurement = detector.data[i]

            feature_measurement.true_time = t-t0
            feature_measurement.seq = i

            for uv, pfw in zip(msg.pixel_location, msg.global_position):
                pixel_feature = PixelFeature()
                pixel_feature.u = uv.u
                pixel_feature.v = uv.v
                pixel_feature.p_f_w[:,0] = [pfw.x, pfw.y, pfw.z]
                feature_measurement.features.append(pixel_feature)

            i+= 1
    detector.data = detector.data[:i]

    return detector

def get_bag(bag_directory,bag_name,topic_name):
    os.chdir(os.environ['HOME']+bag_directory)

    print "Reading " + bag_name
    bag = rosbag.Bag(bag_name)
    num_msgs = bag.get_message_count(topic_name)
    return bag,num_msgs
