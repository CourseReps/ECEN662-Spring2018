import os

import numpy as np

from aims import *
from aims.attkins import AngularRate

import rosbag
from cv_bridge import CvBridge, CvBridgeError

from copy import deepcopy

class Measurement(object):

    def __init__(self,m_time,delay):
        self.true_time = m_time
        self.delay = 0.
        self.seq = 0.

    def __repr__(self):
        str_repr = ""
        attribute_dict = vars(self)
        for key in attribute_dict:
            str_repr += key + ": \n" + str(attribute_dict[key]) + "\n"
        return str_repr

class IMUMeasurement(Measurement):

    def __init__(self,m_time=0.,delay=0.):
        self.angular_rate = AngularRate()
        self.acceleration = np.zeros((3,1))
        super(IMUMeasurement,self).__init__(m_time,delay)

class RangeMeasurement(Measurement):

    def __init__(self,m_time=0.,delay=0.):
        self.range = 0.
        super(RangeMeasurement,self).__init__(m_time,delay)

class GPSMeasurement(Measurement):

    def __init__(self,m_time=0.,delay=0.):
        self.position = np.zeros((3,1))
        super(GPSMeasurement,self).__init__(m_time,delay)

class PixelFeature(object):

    def __init__(self,u=0,v=0):
        self.id = 0
        self.u = u
        self.v = v
        self.p_f_w = np.zeros((3,1))

    def __repr__(self):
        str_repr = ""
        attribute_dict = vars(self)
        for key in attribute_dict:
            str_repr += key + ": " + str(attribute_dict[key]) + " "
        return str_repr

class FeatureMeasurement(Measurement):

    def __init__(self,m_time=0.,delay=0.):
        self.features = []
        super(FeatureMeasurement,self).__init__(m_time,delay)

class ImageMeasurement(Measurement):

    def __init__(self,m_time=0.,delay=0.):
        self.image = None
        super(ImageMeasurement,self).__init__(m_time,delay)

class Sensor(object):

    def __init__(self, length,m_type):
        self.data = [m_type() for i in xrange(length)]
        self.length = length
        self.upcoming_measure_index = 0
        self.current_measure_index = 0

    def add_clock(self,clock):
        """
        Polling requires a reference clock.
        """
        self.clock = clock

    def poll(self):
        """
        Poll the sensor to see if a new measurement has come in since the
        last poll.
        """

        # by default return false
        ret_value = False

        # loop is necessary for the case that multiple measurements come in
        # between calls to poll
        while True:
            # if we haven't already used all of the measurements...
            if self.upcoming_measure_index < self.length:
                # figure out when the next measurement will be available
                upcoming_measure = self.data[self.upcoming_measure_index]
                upcoming_measure_time = upcoming_measure.true_time + upcoming_measure.delay

                if self.clock.now() >= (upcoming_measure_time-0.0000001):
                    # print self.clock.now(), upcoming_measure
                    self.current_measure_index = self.upcoming_measure_index
                    self.upcoming_measure_index += 1
                    ret_value = True
                else:
                    break
            # if we have used all of the measurements...
            else:
                break

        return ret_value

    def reset(self):
        self.upcoming_measure_index = 0
        self.current_measure_index = 0


    def latest(self):
        return self.data[self.current_measure_index]

class Camera:
    def __init__(self,rosbag_directory,bag_name,topic_name,t0):
        self.data = ImageMeasurement()
        self.upcoming_measure_index = 0
        self.rosbag_name = os.path.join(rosbag_directory,bag_name)
        self.topic = topic_name
        self.bag_open = False
        self.t0 = t0


    def add_clock(self,clock):
        """
        Polling requires a reference clock.
        """
        self.clock = clock

    def close(self):
        self.bag.close()
        self.bag = None
        self.message_generator = None

    def poll(self):
        """
        Poll the sensor to see if a new measurement has come in since the
        last poll.
        """

        if not self.bag_open:
            self.bag = rosbag.Bag(os.environ['HOME']+"/"+self.rosbag_name)
            self.length = self.bag.get_message_count(self.topic)
            self.upcoming_measure = ImageMeasurement()
            # image_msg = self.bag.read_messages(self.topic).next().message
            self.message_generator = self.bag.read_messages(self.topic)
            next_msg = self.message_generator.next().message
            self.upcoming_measure.true_time = next_msg.header.stamp.to_sec()- self.t0
            bridge = CvBridge()
            frame = bridge.imgmsg_to_cv2(next_msg, desired_encoding="passthrough")
            self.upcoming_measure.image = frame

        self.bag_open = True

        # by default return false
        ret_value = False

        # loop is necessary for the case that multiple measurements come in
        # between calls to poll
        while True:
            # if we haven't already used all of the measurements...
            if self.upcoming_measure_index < self.length -1:
                # figure out when the next measurement will be available
                upcoming_measure_time = self.upcoming_measure.true_time + self.upcoming_measure.delay

                if self.clock.now() >= (upcoming_measure_time-0.0000001):
                    ret_value = True

                    self.current_measurement = deepcopy(self.upcoming_measure)
                    bridge = CvBridge()
                    next_msg = self.message_generator.next().message
                    frame = bridge.imgmsg_to_cv2(next_msg, desired_encoding="passthrough")
                    self.upcoming_measure.image = frame
                    self.upcoming_measure.true_time = next_msg.header.stamp.to_sec() - self.t0

                    self.upcoming_measure_index += 1

                else:
                    break
            # if we have used all of the measurements...
            else:
                break

        return ret_value

    def latest(self):
        return self.current_measurement

class IMU(Sensor):

    def __init__(self,length):
        super(IMU,self).__init__(length,IMUMeasurement)

class GPS(Sensor):

    def __init__(self,length):
        super(GPS,self).__init__(length,GPSMeasurement)

class RangeFinder(Sensor):

    def __init__(self,length):
        super(RangeFinder,self).__init__(length,RangeMeasurement)

    def to_array(self):
        length = len(self.data)
        time_array = np.zeros(length)
        range_array = np.zeros((1,length))
        for i,range_measure in enumerate(self.data):
            time_array[i] = range_measure.true_time
            range_array[:,i] = range_measure.range

        return time_array,range_array

class FeatureDetector(Sensor):

    def __init__(self,length):
        super(FeatureDetector,self).__init__(length,FeatureMeasurement)
